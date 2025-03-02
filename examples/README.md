# Agent-Based Customer Chatbot Workflow

This file contains example questions for the chatbot, taken from the **"questions.txt"** file, along with **tool calls** and the model's final responses. The following workflow details each step in processing a customer query.

---

## Workflow Overview

1. **Customer Question:**  
   The initial query submitted by the customer.

2. **Initial AI Message & Tool Call:**  
   The model receives the question and decides on processing steps. It **initiates a tool call** with:  
   - **Tool Name & Call ID:** The specific tool invoked and its unique identifier.  
   - **Arguments:** Parameters sent to the tool (e.g., query details or product name).

3. **Tool Response:**  
   The output returned by the tool (e.g., product details, search results, or FAQ answers).

4. **Final Model Response:**  
   The AI synthesizes the tool's output to deliver a clear and concise final answer to the customer.

---

## Example Conversation Flow

### **Example 1: Technical Specifications Query**

- **Customer Question:**  
  *"hi, what are the technical specifications of the phone I bought last time?"*

- **Initial AI Message & Tool Call:**  
  - **Tool:** `product_specifications_retriever`  
  - **Call ID:** `call_aEsBaibym2JSsrlyf04mgwxn`  
  - **Arguments:**  
    - `query: Samsung Galaxy S21`

- **Tool Response:**  
  Technical details for the Samsung Galaxy S21 are returned.

- **Final Model Response:**  
  *"The Samsung Galaxy S21 has the following technical specifications:"*  
  **Display:** 6.2-inch Dynamic AMOLED 2X  
  **Processor:** Exynos 2100  
  **Connectivity:** 5G support  
  **Battery:** 4000mAh with fast charging capabilities  
  **Colors:** Available in several colors  
  **Storage:** Various configurations available  
  *"If you need more specific details, feel free to ask!"*

---

### **Example 2: Inquiry About Other Smartphone Options**

- **Customer Question:**  
  *"what are the other smartphone options?"*

- **Initial AI Message & Tool Call:**  
  - **Tool:** `search_products`  
  - **Call ID:** `call_TvpXL5QwpWdCcHAKlJ2AW1Ok`  
  - **Arguments:**  
    - `category_name: smartphones`  
    - `in_stock: True`

- **Tool Response:**  
  A JSON list of available smartphones, including iPhone 13, Samsung Galaxy S21, Google Pixel 6, and OnePlus 9.

- **Final Model Response:**  
  *"Here are some smartphone options currently available:"*  
  **1. iPhone 13**  
  &nbsp;&nbsp;&nbsp;**Price:** $699.99, **Stock:** 50 units  
  **2. Samsung Galaxy S21**  
  &nbsp;&nbsp;&nbsp;**Price:** $799.99, **Stock:** 27 units  
  **3. Google Pixel 6**  
  &nbsp;&nbsp;&nbsp;**Price:** $599.99, **Stock:** 25 units  
  **4. OnePlus 9**  
  &nbsp;&nbsp;&nbsp;**Price:** $729.99, **Stock:** 40 units  
  *"If you would like more information or purchasing assistance, just let me know!"*

---

### **Example 3: Pros and Cons of Samsung Galaxy**

- **Customer Question:**  
  *"tell me about samsung galaxy, what are pros and cons?"*

- **Initial AI Message & Tool Call:**  
  - **Tool:** `tavily_search_results_json`  
  - **Call ID:** `call_Kx9GNQf1j47ZFqPzneDuiphW`  
  - **Arguments:**  
    - `query: Samsung Galaxy S21 pros and cons`

- **Tool Response:**  
  Pros and cons from multiple review sources (e.g., TechSpot, CNET, PCMag).

- **Final Model Response:**  
  **Pros:**  
  - *Affordable for a flagship device*  
  - *Powerful performance (Snapdragon 888/Exynos 2100)*  
  - *5G connectivity*  
  - *Modern, attractive design*  
  - *Compact size*  

  **Cons:**  
  - *No microSD card slot for expandable storage*  
  - *Lower resolution screen compared to some competitors*  

  *"For more detailed reviews, refer to the provided links."*

---

### **Example 4: Delivery Options Inquiry**

- **Customer Question:**  
  *"what are the delivery options for this smartphone?"*

- **Initial AI Message & Tool Call:**  
  - **Tool:** `retriever_faq`  
  - **Call ID:** `call_FLRsp1jITRwsacnSPlNDJf2h`  
  - **Arguments:**  
    - `query: delivery options for smartphones`

- **Tool Response:**  
  Details regarding Standard Shipping and Express Shipping (including cost and delivery times).

- **Final Model Response:**  
  *"For the Samsung Galaxy S21, the available delivery options are:"*  
  **Standard Shipping:** $5.99, estimated delivery in 3-5 business days  
  **Express Shipping:** $15.99, estimated delivery in 1-2 business days  
  *"Note that options may vary based on destination and product availability."*

---

### **Example 5: Return Policy Inquiry**

- **Customer Question:**  
  *"btw can I return this smartphone after buying and how?"*

- **Initial AI Message & Tool Call:**  
  - **Tool:** `retriever_faq`  
  - **Call ID:** `call_nmE1nmAHlMAPPjRgasHKFS6Q`  
  - **Arguments:**  
    - `query: return policy for smartphones`

- **Tool Response:**  
  Return policy details, including a **30-day return window**, conditions, and customer service contact instructions.

- **Final Model Response:**  
  *"Yes, you can return the Samsung Galaxy S21 after purchase."*  
  **Return Policy:**  
  - **Return Period:** 30 days  
  - **Condition:** Item must be in its original condition and packaging  

  **How to Initiate a Return:**  
  1. **Contact Customer Service**  
  2. **Follow the Provided Instructions**  

  *"Feel free to ask if you need further assistance with the return process."*

---

### **Example 6: Purchase Request with Express Shipping**

- **Customer Question:**  
  *"okey I liked it, I wanna buy this one, please buy it for me with express shipping"*

- **Initial AI Message & Tool Call:**  
  - **Tool:** `easy_buy_product`  
  - **Call ID:** `call_kdJTJ2aLJEPf5bAitswMRGFh`  
  - **Arguments:**  
    - `product_name: Samsung Galaxy S21`  
    - `quantity: 1`  
    - `shipping_method: Express Shipping`

- **Tool Response:**  
  Confirmation of the purchase process initiation.

- **Final Model Response:**  
  The final confirmation message with purchase details is delivered to the customer.

---

### **Example 7: Order History and Delivery Status Inquiry**

- **Customer Question:**  
  *"what was my last purchase and what is the estimated delivery date?"*

- **Initial AI Message & Tool Calls:**  
  - **Tool 1:** `fetch_shipping_status`  
    - **Call ID:** `call_6Y3DX9gSAsEjzVd07zBCdLyH`  
    - **Arguments:**  
      - `customer_id: 1, order_id: 6`
  - **Tool 2:** `fetch_user_order_information`  
    - **Call ID:** `call_ARsMssMvsC0bQnI7ClcGZfCd`

- **Tool Response:**  
  The latest order details (Order ID 6) including shipping method and estimated delivery date.

- **Final Model Response:**  
  *"Your last purchase was the **Samsung Galaxy S21**. Here are the details:"*  
  **Order ID:** 6  
  **Order Date:** January 6, 2025  
  **Shipping Method:** Express Shipping  
  **Shipping Cost:** $15.99  
  **Estimated Delivery Date:** January 8, 2025  
  *"If you have any more questions or need further assistance, feel free to ask!"*

---

## **Summary**

This workflow demonstrates how the **agent-based customer chatbot** processes queries by:

- **Receiving the customer question** (sourced from **"questions.txt"**)
- **Executing tool calls** with clear parameters and unique IDs
- **Presenting tool responses** (such as product specifications, search results, or FAQ answers)
- **Synthesizing the final answer** for the customer in an easy-to-understand format

This transparent process allows users to see exactly **what happens behind the scenes**—from **tool invocation** to the **final response**.

