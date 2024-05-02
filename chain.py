from config import *

def setup_data_chain(llm_data,vectorstore,memory):
    data_chain = ConversationalRetrievalChain.from_llm(llm=llm_data,
                                                       retriever = vectorstore.as_retriever(search_type="mmr"),
                                                       verbose=True,
                                                       memory = memory)
    return data_chain

def setup_stage_analyzer_inception_chain(llm_checker, memory):
    stage_analyze_inceptionr_template = """You are a sales assistant, your job is:
        1. Use the conversation history at the end.
        2. Then, determine the next stage in the sales conversation by choosing from 1 of the options below.
        3. Ensure compliance with requirements.
        ```
        Requirements:
        1. The best guess of what stage should the conversation continue from i of 7 options.
        2. Only answer numerically.
        3. The answer needs to be one number only, no words.
        4. If there is no conversation history, output "1".
        '''
        Options:
        1. Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.
        2. Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.
        3. Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.
        4. Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
        5. Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.
        6. Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.
        7. Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.
        8: End conversation: The prospect has to leave, the prospect is not interested, or the next steps were already determined by the sales agent.
        '''
        """

    stage_analyze_inceptionr_template_human = """Conversation history:
    {chat_history}"""


    stage_analyzer_inception_prompt = ChatPromptTemplate.from_messages(
        [
            ( "system", stage_analyze_inceptionr_template),
            ("human", stage_analyze_inceptionr_template_human)
        ]
    )

    stage_analyzer_inception_chain = (
        {"chat_history": lambda x: memory.load_memory_variables(x)["chat_history"]}
        | stage_analyzer_inception_prompt
        | llm_checker
        | StrOutputParser()
    )

    return stage_analyzer_inception_chain

def setup_conversation_chain(llm_chat, memory):
    template = """Never forget your name is {salesperson_name}. You work as a {salesperson_role}.
    You work at company named {company_name}. {company_name}'s business is the following: {company_business}.
    Company values are the following. {company_values}
    You are contacting a potential prospect in order to {conversation_purpose}
    Your means of contacting the prospect is {conversation_type}

    If you're asked about where you got the user's contact information, say that you got it from public records.
    Keep your responses in short length to retain the user's attention.
    Start the conversation by just a greeting and how is the prospect doing without pitching in your first turn.

    Ticket information:
    {data}"""

    human_template ="""{input}"""

    prompt = ChatPromptTemplate.from_messages(
            [
                ( "system", template),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human_template)
            ]
        )
    conversation_chain =(
            {
            "data": lambda x : x["data"],
            "input":lambda x : x["input"],
            "salesperson_name": lambda x : x["salesperson_name"],
            "salesperson_role": lambda x : x["salesperson_role"],
            "company_name": lambda x : x["company_name"],
            "company_business": lambda x : x["company_business"],
            "company_values": lambda x : x["company_values"],
            "conversation_purpose": lambda x : x["conversation_purpose"],
            "conversation_type": lambda x : x["conversation_type"],
            }
            | prompt
            | llm_chat
            | StrOutputParser()
            
        )

    conversation_chain_with_chat_history = RunnableWithMessageHistory(
            conversation_chain,
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: memory.chat_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    return conversation_chain_with_chat_history

def get_product_information(data_chain,input : str) ->str:
    result = data_chain.invoke({"question": input})
    return result["answer"]

def get_conversation_Stage(stage_analyzer_inception_chain) -> str:
    conversation_stages = {
    "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
    "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
    "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
    "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
    "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
    "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
    "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
    "8": "End conversation: The prospect has to leave, the prospect is not interested, or the next steps were already determined by the sales agent.",
    }
    
    result = stage_analyzer_inception_chain.invoke(" ")
    result = result.strip()
    index = re.findall(r'\d+', result)[0]
    return conversation_stages[index]

def invoke_chain(input: str, stage_analyzer_inception_chain, data_chain, conversation_chain)->str:
    conversation_stages = get_conversation_Stage(stage_analyzer_inception_chain)
    print("Conversation Stages:\n"+conversation_stages+"\n---")
    data = get_product_information(data_chain,input)
    print("Data:\n"+data+"\n---")
    
    return conversation_chain.invoke({
    "input":input,
    "data": data,
    "salesperson_name": "Lan Huong",
    "salesperson_role": "Ticker seller",
    "company_name": "Đảo Ký Ức Hội An",
    "company_business": "Đảo Ký Ức Hội An là quần thể du lịch – nghỉ dưỡng – văn hóa nằm kề cận và là một phần không thể tách rời khỏi phố cổ Hội An, là đơn vị tiên phong trong xu hướng du lịch văn hóa tại Việt Nam.",
    "company_values": "Đảo Ký Ức Hội An kiến tạo không gian du lịch xanh, cùng du lịch địa phương hướng đến hệ sinh thái du lịch văn hóa bền vững. Show diễn Ký Ức Hội An - tái hiện lại 400 năm Hội An thăng trầm, với 5 màn trình diễn ấn tượng của gần 500 diễn viên chuyên nghiệp, trên sân khấu rộng 25.000m2 giữa sông Hoài. Công viên Văn hóa chủ đề đầu tiên tại Việt Nam - Ấn Tượng Hội An, sử dụng sự đa dạng trong kiến trúc và nghệ thuật, phản ánh đời sống 1 cảng thị quốc tế sầm uất.",
    "conversation_purpose": "Find out if they want to experience an exciting trip by purchasing a tour ticket.",
    "conversation_type": "message", },
    config={"configurable": {"session_id": "sale"}}
    )