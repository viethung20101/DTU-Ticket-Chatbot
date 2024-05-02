from config import *

def format_data():
    with psycopg2.connect(CONN_STR) as conn:
        data = pd.read_sql("""SELECT id, name, price, overview, included_items, meeting_point, expectations, additional_info, cancellation_policy 
	                          FROM public.tickets;""", conn)

    data_format = ""
    for i in range(data.shape[0]):
        data_format += f"""
            Vé thứ: {i + 1}
            Tên vé: {data.iloc[i, 1]}
            Thông tin vé: {data.iloc[i, 3]}
            {data.iloc[i, 4]}
            {data.iloc[i, 5]}
            {data.iloc[i, 6]}
            {data.iloc[i, 7]}
            {data.iloc[i, 8]}
            Giá vé {data.iloc[i, 1]}: {int(data.iloc[i, 2])} VND
            """
    return data_format

def create_vector_store(content, embeddings, chunk_size = 1024, chunk_overlap = 256):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    splits = text_splitter.split_text(content)
    vectorstore = Qdrant.from_texts(
        splits,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="ticket",
    )
    
    return vectorstore