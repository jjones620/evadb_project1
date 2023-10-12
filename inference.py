from gpt4all import GPT4All

import evadb
import numpy as np

class Model:
    def __init__(self):
        self.cursor = evadb.connect().cursor()
        self.inference_table = "InferenceTable"
        self.inference_feat_table = "InferenceFeatTable"

        self.setup()
        self.llm = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")

    def setup(self):
        


        print("Setup Function")

        Text_feat_function_query = f"""CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
                IMPL  './sentence_feature_extractor.py';
                """

        self.cursor.query("DROP FUNCTION IF EXISTS SentenceFeatureExtractor;").execute()
        self.cursor.query(Text_feat_function_query).execute()

        self.cursor.query("DROP FUNCTION IF EXISTS Similarity;").execute()
        Similarity_function_query = """CREATE FUNCTION Similarity
                        INPUT (Frame_Array_Open NDARRAY UINT8(3, ANYDIM, ANYDIM),
                            Frame_Array_Base NDARRAY UINT8(3, ANYDIM, ANYDIM),
                            Feature_Extractor_Name TEXT(100))
                        OUTPUT (distance FLOAT(32, 7))
                        TYPE NdarrayFunction
                        IMPL './similarity.py'"""
        
        self.cursor.query(Similarity_function_query).execute()

        self.cursor.query(f"DROP TABLE IF EXISTS {self.inference_table};").execute()
        self.cursor.query(f"DROP TABLE IF EXISTS {self.inference_feat_table};").execute()


        print("Create table")

        self.cursor.query(f"CREATE TABLE {self.inference_table} (id INTEGER, conversation TEXT(1000));").execute()

    def ask_question(self, query: str) -> str: 

        np_array = np.array(list(query))

        res_batch = self.cursor.query(
            f"""SELECT conversation FROM {self.inference_table}
            ORDER BY Similarity(SentenceFeatureExtractor('{query}'), conversation)
            LIMIT 5;"""
        ).execute()


        context_list = []
        for i in range(len(res_batch)):
            previous_question = res_batch.frames[f"{self.inference_table.lower()}.conversation"][i]
            context_list.append(previous_question)
        context = "\n".join(context_list)


        query = f"""If the context is not relevant, please answer the question by using your own knowledge about the topic.
        
        {context}
        
        Question : {query}"""

        full_response = self.llm.generate(query)

        self.insert_query(query, full_response)

        return full_response
    def insert_query(self, query, response):
        find_latest_index_query = f"SELECT id FROM {self.inference_table} ORDER BY id DESC LIMIT 1;"
        
        res = self.cursor.query(find_latest_index_query).execute()
        latest_index = -1
        if len(res.columns) > 0:
            latest_index = res.frames[f"{self.inference_table.lower()}.id"][0]
        else:
            latest_index = 0
        
        insert_query = f"INSERT INTO {self.inference_table} (id, conversation) VALUES ({latest_index + 1}, '{query + response}');"

        # insert_query = f"INSERT INTO {self.inference_table} VALUES ({latest_index + 1}, '{query + response}');"
        self.cursor.query(insert_query).execute()
