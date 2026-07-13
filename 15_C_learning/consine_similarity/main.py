from sentence_transformers import SentenceTransformer
# from sklearn.module.pairwises import cosine_similarity 
from math import sqrt

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "what is the name of cat"
sentence2 =  "what is the capital of india"

vector1 = model.encode(sentence1);
vector2 = model.encode(sentence2);

dot_product = 0 ;

for i in range(len(vector1)):
    dot_product += vector1[i]*vector2[i];

modA = 0; 
modB = 0;

for i in range(len(vector1)):
    modA += vector1[i]**2;
    modB += vector2[i]**2;

modA = sqrt(modA);
modB = sqrt(modB);

cosine_similarity = dot_product/(modA*modB);

print(cosine_similarity);

