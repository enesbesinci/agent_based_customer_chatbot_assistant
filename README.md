# Agent Based Customer Chatbot Assistant

### **Customer Shopping Assistant - Project Overview**  

This project introduces an **LLM-powered agent-based chatbot** designed to enhance the online shopping experience. The system efficiently handles customer interactions by providing real-time product details, performing filtered searches, comparing products, managing purchases, and tracking orders.  

#### **Key Objectives**  
- **Seamless Customer Assistance:** Ensures 24/7 availability, delivering instant and accurate responses to customer inquiries.  
- **Operational Efficiency:** Automates routine customer support tasks, reducing the need for human intervention.  
- **Enhanced Shopping Experience:** Provides personalized recommendations and real-time order tracking.  

#### **Technology Stack**  
- **LLM Model:** gpt-4o-mini from OpenAI 
- **Embedding Model:** text-embedding-3-small from OpenAI   
- **Database:** SQLite  
- **Vector Storage:** Chroma  
- **Re-Ranking Model:** rerank-english-v3.0 from Cohere   
- **Agent & Workflow Management:** LangChain, LangGraph  
- **User Interface:** Streamlit  

#### **System Architecture**  
- **Data Handling & Storage:** A structured **SQL database** has been developed to manage product information, orders, and customer interactions.  
- **Functional Components:**  
  - **SQL-Based Agents:** Execute structured queries for product retrieval, order tracking, and purchase management.  
  - **RAG-Based Information Retrieval:** Utilizes a **retrieval-augmented generation (RAG)** approach for providing product specifications, customer reviews, and policy details.  
  - **Web Search Integration:** Enables real-time access to external information when necessary.  
- **Content Moderation & Guardrails:**  
  - **LLM-Based Input Filtering:** Ensures chatbot interactions remain relevant and appropriate.  
  - **Security Mechanisms:** Implemented to prevent harmful or misleading content generation.  

This project delivers a **scalable, intelligent, and secure AI-driven shopping assistant**, enhancing customer engagement and optimizing e-commerce support operations. For further details, please refer to the project documentation.


