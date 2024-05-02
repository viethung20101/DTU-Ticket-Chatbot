from __init import *
from data import *

data_format = format_data()

vectorstore = create_vector_store(data_format, embeddings)