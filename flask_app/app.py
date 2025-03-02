# library imports
import os
import sqlite3
import random
from datetime import datetime, timedelta
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_cohere import CohereRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from datetime import datetime
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Optional, List
from langchain_core.runnables import RunnableConfig
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# create the flask_app
app = Flask(__name__)

# load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # your openai api key

# create an llm from langchain_opeanai to use in the agent
llm = ChatOpenAI(model="gpt-4o-mini",
                 temperature=0,
                 openai_api_key=OPENAI_API_KEY)

# create an embedding model from langchain_openai to use in the agent
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              openai_api_key=OPENAI_API_KEY)


# sql database: e-commerce-data.sqlite
local_file = "your sql database path that was created in the "creating_example_data.ipynb""

# tools

# fetch user order information

@tool
def fetch_user_order_information(config: RunnableConfig) -> list[dict]:

    """
    Fetches all orders for a specific user along with product details.

    Retrieves the user's order information, including order ID, order date, product name, and quantity.
    
    Args:
        config (RunnableConfig): Configuration object containing the user ID.

    Returns:
        list[dict]: A list of dictionaries containing order and product details for the user.
    """

    if config is None:
        raise ValueError("Config is not provided.")
    
    configuration = config.get("configurable", {})
    CUSTOMER_ID = configuration.get("CUSTOMER_ID", None)
    if not CUSTOMER_ID:
        raise ValueError("No user CUSTOMER_ID provided.")

    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    query = """
    SELECT
	    C.CUSTOMER_ID,
	    C.FIRST_NAME,
	    C.LAST_NAME,
	    O.ORDER_ID,
	    O.ORDER_DATE,
	    P.PRODUCT_NAME,
	    OD.QUANTITY,
	    P.PRICE
    FROM
        Customers C
    INNER JOIN
	    Orders O ON O.CUSTOMER_ID = C.CUSTOMER_ID
    INNER JOIN
	    OrderDetails OD ON OD.ORDER_ID = O.ORDER_ID
    INNER JOIN
	    Products P ON P.PRODUCT_ID = OD.PRODUCT_ID
    WHERE
	    C.CUSTOMER_ID = ?
    """
    cursor.execute(query, (CUSTOMER_ID,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    
    return results

# search for products

@tool
def search_products(
    category_name: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    product_name: Optional[str] = None,
    in_stock: Optional[bool] = None,
    limit: int = 20
) -> List[dict]:
    """Search for products in the database and filter based on product name, category name, price range and stock availability."""

    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    query = """
    SELECT
        P.PRODUCT_ID,
        C.CATEGORY_NAME,
        P.PRODUCT_NAME,
        P.PRICE,
        P.STOCK
    FROM Products P
    INNER JOIN Categories C ON C.CATEGORY_ID = P.CATEGORY_ID
    WHERE 1 = 1
    """
    params = []

    if category_name:
        query += " AND LOWER(C.CATEGORY_NAME) = LOWER(?)"
        params.append(category_name)

    if min_price:
        query += " AND P.PRICE >= ?"
        params.append(min_price)

    if max_price:
        query += " AND P.PRICE <= ?"
        params.append(max_price)

    if product_name:
        query += " AND P.PRODUCT_NAME LIKE ?"
        params.append(f"%{product_name}%")

    if in_stock is not None:
        query += " AND P.STOCK > 0" if in_stock else " AND P.STOCK = 0"

    query += " LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

# product recommendation tool

@tool
def product_recommendations(config: RunnableConfig) -> dict:
    """
    Fetches all orders for a specific user along with product details and recommends other products in the same category.
    
    Retrieves the user's order information, including order ID, order date, product name, and quantity.
    Additionally, it recommends other products from the same categories the user has purchased from.
    
    Args:
        config (RunnableConfig): Configuration object containing the user ID.
    
    Returns:
        dict: A dictionary containing the user's order details and product recommendations.
    """

    if config is None:
        raise ValueError("Config is not provided.")
    
    configuration = config.get("configurable", {})
    CUSTOMER_ID = configuration.get("CUSTOMER_ID", None)
    if not CUSTOMER_ID:
        raise ValueError("No user CUSTOMER_ID provided.")
    
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    # Fetch user's order information
    order_query = """
    SELECT
        C.CUSTOMER_ID,
        C.FIRST_NAME,
        C.LAST_NAME,
        O.ORDER_ID,
        O.ORDER_DATE,
        P.PRODUCT_NAME,
        P.CATEGORY_ID,
        OD.QUANTITY,
        P.PRICE
    FROM
        Customers C
    INNER JOIN
        Orders O ON O.CUSTOMER_ID = C.CUSTOMER_ID
    INNER JOIN
        OrderDetails OD ON OD.ORDER_ID = O.ORDER_ID
    INNER JOIN
        Products P ON P.PRODUCT_ID = OD.PRODUCT_ID
    WHERE
        C.CUSTOMER_ID = ?
    """
    cursor.execute(order_query, (CUSTOMER_ID,))
    order_rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    order_details = [dict(zip(column_names, row)) for row in order_rows]

    # Fetch product recommendations based on the categories of purchased products
    category_ids = tuple(set(row['CATEGORY_ID'] for row in order_details))
    if category_ids:
        recommendation_query = """
        SELECT PRODUCT_NAME, PRICE 
        FROM Products 
        WHERE CATEGORY_ID IN ({seq}) 
        AND PRODUCT_ID NOT IN (
            SELECT OD.PRODUCT_ID
            FROM OrderDetails OD
            INNER JOIN Orders O ON OD.ORDER_ID = O.ORDER_ID
            WHERE O.CUSTOMER_ID = ?
        )
        LIMIT 5
        """.format(seq=','.join(['?'] * len(category_ids)))

        cursor.execute(recommendation_query, category_ids + (CUSTOMER_ID,))
        recommendations = cursor.fetchall()
        recommendations = [{"PRODUCT_NAME": row[0], "PRICE": row[1]} for row in recommendations]
    else:
        recommendations = []

    cursor.close()
    conn.close()

    return {"order_details": order_details, "recommendations": recommendations}

# fetch shipping status

@tool
def fetch_shipping_status(
    customer_id: Optional[int] = None,
    order_id: Optional[int] = None
) -> List[dict]:
    """
    Fetches the shipping status for a specific customer or order.

    Args:
        customer_id (Optional[int]): The customer ID to fetch shipping details for.
        order_id (Optional[int]): The order ID to fetch shipping details for.

    Returns:
        List[dict]: A list of dictionaries containing shipping information.
    """

    if customer_id is None and order_id is None:
        raise ValueError("Either customer_id or order_id must be provided.")

    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    query = """
    SELECT 
        O.ORDER_ID,
        C.FIRST_NAME,
        C.LAST_NAME,
        S.SHIPPING_METHOD,
        S.SHIPPING_COST,
        S.ESTIMATED_DELIVERY
    FROM Orders O
    INNER JOIN Customers C ON O.CUSTOMER_ID = C.CUSTOMER_ID
    INNER JOIN Shipping S ON O.SHIPPING_ID = S.SHIPPING_ID
    WHERE 1 = 1
    """
    params = []

    if customer_id:
        query += " AND C.CUSTOMER_ID = ?"
        params.append(customer_id)

    if order_id:
        query += " AND O.ORDER_ID = ?"
        params.append(order_id)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    column_names = [column[0] for column in cursor.description]
    shipping_details = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return shipping_details

# buying product

@tool
def easy_buy_product(config: RunnableConfig,
                     product_name: str,
                     quantity: int = 1,
                     shipping_method: str = "Standart Shipping") -> dict:
    """
    Simplifies the product purchase process by allowing customers to specify the quantity and shipping method.
    The customer ID is automatically retrieved from the config file.
    The payment and shipping details are auto-generated, and the payment date is set to the current date.
    The shipping_id is set as the maximum existing shipping_id + 1.

    Args:
        product_name (str): The name of the product to purchase.
        quantity (int, optional): The number of products to purchase. Defaults to 1.
        shipping_method (str, optional): The method of shipping. Defaults to "Standart Shipping".

    Returns:
        dict: A dictionary containing the order details and status.
    """
    # Customer ID is retrieved from the config file
    customer_id = config['configurable']['CUSTOMER_ID']

    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    # Fetch product details based on the product name
    cursor.execute("SELECT PRODUCT_ID, STOCK, PRICE FROM Products WHERE PRODUCT_NAME = ?", (product_name,))
    product = cursor.fetchone()

    if not product:
        conn.close()
        return {"status": "error", "message": f"Product '{product_name}' not found."}

    product_id, stock, price = product

    # Check stock availability
    if stock < quantity:
        conn.close()
        return {"status": "error", "message": f"Insufficient stock for '{product_name}'."}

    # Determine shipping details
    if shipping_method == "Express Shipping":
        shipping_cost = 15.99
        estimated_delivery = datetime.now() + timedelta(days=random.randint(1, 2))  # 1-2 gün
    else:
        shipping_cost = 5.99
        estimated_delivery = datetime.now() + timedelta(days=random.randint(3, 5))  # 3-5 gün

    payment_method = "Credit Card"
    payment_date = datetime.now().strftime("%Y-%m-%d")  # Current date as payment date
    amount = price * quantity + shipping_cost  # Calculate total amount including shipping cost

    # Fetch the maximum shipping_id and set a new one as max + 1
    cursor.execute("SELECT MAX(SHIPPING_ID) FROM Orders")
    max_shipping_id = cursor.fetchone()[0]
    shipping_id = max_shipping_id + 1 if max_shipping_id else 1  # If no shipping_id exists, start from 1

    # Insert shipping information
    cursor.execute("INSERT INTO Shipping (SHIPPING_ID, SHIPPING_METHOD, SHIPPING_COST, ESTIMATED_DELIVERY) VALUES (?, ?, ?, ?)",
                   (shipping_id, shipping_method, shipping_cost, estimated_delivery.strftime("%Y-%m-%d")))

    # Insert order
    cursor.execute("INSERT INTO Orders (CUSTOMER_ID, ORDER_DATE, SHIPPING_ID, PAYMENT_ID) VALUES (?, date('now'), ?, ?)",
                   (customer_id, shipping_id, None))  # Payment_ID will be updated later
    order_id = cursor.lastrowid

    # Insert order details
    cursor.execute("INSERT INTO OrderDetails (ORDER_ID, PRODUCT_ID, QUANTITY, UNIT_PRICE) VALUES (?, ?, ?, ?)",
                   (order_id, product_id, quantity, price))

    # Insert payment
    cursor.execute("INSERT INTO Payments (PAYMENT_METHOD, PAYMENT_DATE, AMOUNT) VALUES (?, ?, ?)",
                   (payment_method, payment_date, amount))
    payment_id = cursor.lastrowid

    # Update the order with payment ID
    cursor.execute("UPDATE Orders SET PAYMENT_ID = ? WHERE ORDER_ID = ?", (payment_id, order_id))

    # Update product stock
    cursor.execute("UPDATE Products SET STOCK = STOCK - ? WHERE PRODUCT_ID = ?", (quantity, product_id))

    # Commit and close connection
    conn.commit()
    cursor.close()
    conn.close()

    return {"status": "success", "order_id": order_id, "message": f"Successfully purchased {quantity} of '{product_name}' with {shipping_method}."}


# search for technical specifications of products

loader_products = TextLoader("Path to the products.txt file")
products_docs = loader_products.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                               chunk_overlap=64,
                                               add_start_index=True)

all_splits_products = text_splitter.split_documents(products_docs)

# create a semantic-search based retriever
vectorstore_products = Chroma.from_documents(documents=all_splits_products,
                                             collection_name="products-specifications-chroma",
                                             embedding=embeddings)


vectorstore_retriever_products = vectorstore_products.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# create a bm-25 retriever
bm25_retriever_products = BM25Retriever.from_documents(all_splits_products)

# initialize the ensemble retriever
ensemble_retriever_products = EnsembleRetriever(
    retrievers=[bm25_retriever_products, vectorstore_retriever_products], weights=[0.25, 0.75])

# add a re-ranker using CohereRerank
compressor = CohereRerank(model="rerank-english-v3.0")

product_specifications_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever_products)

# create the tool
tool_retriever_technical_specifications = create_retriever_tool(
    product_specifications_retriever,
    "product_specifications_retriever",
    "Fetches and returns product specifications for available items, including detailed technical information like processor, screen size, memory, and more.",
)

# search for comments


loader_comments = TextLoader("path to the product_comments.txt file")
product_comments_doc = loader_comments.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                               chunk_overlap=24)

texts_comments = text_splitter.split_documents(product_comments_doc)

vectorstore_products_comments = Chroma.from_documents(documents=texts_comments,
                                                      collection_name="products-comments-chroma",
                                                      embedding=embeddings)


retriever_product_comments = vectorstore_products_comments.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# create a bm-25 retriever
bm25_retriever_products_comments = BM25Retriever.from_documents(texts_comments)

# initialize the ensemble retriever
ensemble_retriever_products_comments = EnsembleRetriever(
    retrievers=[bm25_retriever_products_comments, retriever_product_comments], weights=[0.25, 0.75])

# add a re-ranker using CohereRerank
compressor = CohereRerank(model="rerank-english-v3.0")

product_comment_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever_products_comments)

tool_retriever_product_comments = create_retriever_tool(
    product_comment_retriever,
    "product_comments_retriever",
    "Fetches and returns customer comments for various products, including positive, negative, and neutral reviews.",
)

# search for company policies

# file path

loader_faq = TextLoader("path to the faq.txt file")
faq_documents = loader_faq.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                               chunk_overlap=64,
                                               add_start_index=True)

all_splits = text_splitter.split_documents(faq_documents)

# create a semantic-search based retriever
vectorstore = Chroma.from_documents(documents=all_splits,
                                    collection_name="faq-chroma",
                                    embedding=embeddings)

vectorstore_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# create a bm-25 retriever
bm25_retriever = BM25Retriever.from_documents(all_splits)

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore_retriever], weights=[0.2, 0.8])

# add a re-ranker using CohereRerank
compressor = CohereRerank(model="rerank-english-v3.0")

retriever_faq = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever)

# Create the retriever tool
tool_retriever_faq = create_retriever_tool(
    retriever_faq,
    "retriever_faq",
    "Fetches and returns frequently asked questions and their answers, including details on purchasing, shipping, company policies, returns, and more."
)

# tavily search tool

tavily_tool = TavilySearchResults(max_results=3)

# Input and Output Guardrails

# Input Guardrail

# Data model
class EvaluateAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_question_evaluator = llm.with_structured_output(EvaluateAnswer)

# Prompt
system = """
You are an AI assistant working for an e-commerce company selling technological products and designed to evaluate and score customer questions. Your task is to evaluate whether the customer question is related to the following topics. If the question is related to the topics below, you score “yes”. Otherwise score it as “no”.

Be careful, some questions may be follow-up questions, use the context of the conversation, chat history, to evaluate such questions.

Related Topics:
- questions about customer's purchases
- questions about delivery tracking
- questions about products
- frequently asked questions such as sales, returns, shipping, payment
- product recommendations
- product purchase requests
"""
evaluate_question = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the user question. \n\n {question} \n Here is the chat history {chat_history} \n\n Check the user question.",
        ),
    ]
)

question_evaluator = evaluate_question | structured_llm_question_evaluator

# Output Guardrail

# Data model
class EvaluateAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_response_evaluator = llm.with_structured_output(EvaluateAnswer)

# Prompt
system = """
You are an AI evaluator tasked with reviewing generated responses to ensure they adhere to ethical standards. Your job is to check whether the responses contain any statements that:

1. **Violate Human Rights**: Look for expressions that glorify violence, racism, sexism, or any form of discrimination.
2. **Mention Competitors**: Ensure that the response does not include any references to the following competitor companies: 
   - Trendyol
   - HepsiBurada
   - Amazon
   - MediaMarkt
   - Vatan Computer
   - n11
   - Pazarama
   - Morhipo
   - Teknosa

Based on your evaluation, provide a binary score:

- Return "yes" if the response contains any of the above inappropriate statements or mentions any of the competitor companies.
- Return "no" if the response is free from such statements and does not mention any competitor companies.

Example Responses:
1. "This product is available at Trendyol." → **yes**
2. "Iphone 15 has a 50 megapixel camera." → **no**
3. "Our service is better than HepsiBurada." → **yes**
4. "I support diversity and inclusion." → **no**
5. "You can find great deals on Amazon." → **yes**
"""
evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the generated response. \n\n {generation} \n Check the generated response.",
        ),
    ]
)

answer_evaluator = evaluate_prompt | structured_llm_response_evaluator

# nodes for input and output guardarails

def warning_answer(state):
    """
    Provides a warning response when the question cannot be answered.

    Args:
        state (dict): The current graph state

    Returns:
        dict: A dictionary containing the warning message as generation.
    """

    return {"messages":[AIMessage(content="I'm sorry, I can't help you with that. How else may I assist you?")]}



def input_guardrails(state):
    """
    It evaluates the question before pass to the chatbot.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict):
    """
    last_message  = state["messages"][-1]

    chat_history = state["messages"][:-1]

    is_question_related = question_evaluator.invoke({"question": last_message,
                                                     "chat_history":chat_history})
    if is_question_related.binary_score == "yes":
        return "yes"
    if is_question_related.binary_score == "no":
        return "no"
    

def output_guardrail(state):
    """
    It evaluates the generated response before it is shown to the user.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict):
    """
    generated_answer = state["messages"][-1]

    response_check = answer_evaluator.invoke({"generation": generated_answer})
    if response_check.binary_score == "yes":
        return {"messages":[AIMessage(content="I'm sorry, I can't help you with that.")]}
    if response_check.binary_score == "no":
        return {"messages": generated_answer}
    
# Utilities

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n Please Fix Your Mistakes!",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


# State

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for an e-commerce company. "
            "Use the provided tools to assist customers with their orders, questions on products, shipping tracking and company policies etc."
            "Be persistent in your searches and expand your query bounds if the first search yields no results. "
            "If a search comes up empty, continue expanding your search until you find the information needed."
            "\n\nCurrent user:\n\n{user_info}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


# read-only tools (such as retrievers) don't need a user confirmation to use
safe_tools = [fetch_user_order_information,
              search_products,
              tavily_tool,
              tool_retriever_faq,
              tool_retriever_technical_specifications,
              product_recommendations,
              fetch_shipping_status]

# sensetive-tools need a user confirmation to use
sensitive_tools = [easy_buy_product]

sensitive_tool_names = {t.name for t in sensitive_tools}
# Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
part_3_assistant_runnable = assistant_prompt | llm.bind_tools(
    safe_tools + sensitive_tools
)

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": fetch_user_order_information.invoke({})}


# NEW: The fetch_user_order_information node runs first, meaning our assistant can see the customer's purchase information without
# having to take an action
builder.add_node("output_guardrail", output_guardrail)
builder.add_node("warning_answer", warning_answer)
builder.add_node("fetch_user_info", user_info)
builder.add_node("assistant", Assistant(part_3_assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
builder.add_node(
    "sensitive_tools", create_tool_node_with_fallback(sensitive_tools)
)
# Define logic
builder.add_conditional_edges(
    START,
    input_guardrails,
    {
        "yes": "fetch_user_info",
        "no": "warning_answer",
    },
)
builder.add_edge("fetch_user_info", "assistant")
builder.add_edge("warning_answer",END)

def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "output_guardrail"]:
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return "output_guardrail"
    ai_message = state["messages"][-1]
    # This assumes single tool calls. To handle parallel tool calling, you'd want to
    # use an ANY condition
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"


builder.add_conditional_edges(
    "assistant",
    route_tools,
)
builder.add_edge("output_guardrail", END)

builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")


memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    # NEW: The graph will always halt before executing the "tools" node.
    # The user can approve or reject (or even alter the request) before
    # the assistant continues
    interrupt_before=["sensitive_tools"], 
)



# FLASK #

@app.route("/chat", methods=["POST"])
def chat():
    config = {
            "configurable": {
                "CUSTOMER_ID": 1,
                "thread_id": 1,
            }
        }
    
    query = request.json.get("query")
    if query:
        response = graph.invoke(
            {"messages": ("user", query)}, config)
        
        answer = response["messages"][-1].content

        return jsonify({"answer": answer})

    return jsonify({"error": "Please ask a question!"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2121)
