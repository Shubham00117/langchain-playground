from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# 1. Large High-Contrast Paragraphs (Detailed content simulation)
text = """
The legacy of the Roman Empire is a foundational pillar of Western civilization, encompassing an immense array of achievements in governance, engineering, and the arts that continue to resonate through the centuries. From the development of advanced hydraulic concrete that allowed for massive structures like the Pantheon to remain standing for two millennia, to the construction of a vast network of roads that unified distant provinces, the Romans were masters of the physical world. Their legal system, particularly the codification of civil law under Justinian, served as the blueprint for many modern legal frameworks across Europe and the Americas. Historians often engage in rigorous debates regarding the complex socioeconomic and political factors—ranging from internal corruption and economic hyperinflation to the relentless pressure of external migrations—that ultimately precipitated the fall of the Western Roman Empire in 476 AD, marking the transition from classical antiquity to the Middle Ages.

In the realm of organic chemistry, the covalent bond represents a fundamental interaction where two atoms achieve a stable electronic configuration by sharing one or more pairs of valence electrons. This sharing is driven by the electromagnetic attraction between the positively charged nuclei and the negatively charged electron clouds. Carbon atoms are particularly notable in this field due to their tetravalency, meaning they can form four stable covalent bonds with a wide variety of other elements, including hydrogen, oxygen, and nitrogen. This unique bonding capability allows for the creation of incredibly complex and diverse macromolecules, such as long-chain hydrocarbons, intricate ring structures, and the vital polymers that constitute the very fabric of life, including DNA, RNA, and proteins. Understanding these molecular interactions is crucial for fields ranging from pharmacology and drug design to materials science and environmental engineering.

A comprehensive and sustainable strength training program is built upon the principle of progressive overload, which necessitates the gradual increase of stress placed upon the body during exercise to stimulate physiological adaptation. This typically involves performing multiple sets of multi-joint, compound exercises—such as the barbell squat, the conventional deadlift, and the flat bench press—which recruit multiple muscle groups and central nervous system pathways simultaneously. To maximize hypertrophic muscle growth and neurological strength gains, athletes must not only focus on the mechanics of lifting but also prioritize meticulous recovery protocols, including adequate sleep quality and the consumption of a nutrient-dense diet. This diet must be particularly high in essential amino acids and complete protein sources to support muscle protein synthesis and repair. Furthermore, monitoring training volume and intensity is essential to prevent the onset of overtraining syndrome and ensure long-term athletic longevity.
"""

# 2. Initialize Embeddings & Chunker with 70th Percentile Threshold
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chunker = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.2 # Robust threshold for large, detailed paragraphs
)

# 4. Perform the splitting
# The chunker should detect the shift from Space to Cooking, and then to Finance
print("Performing semantic chunking...")
chunks = chunker.create_documents([text])

# 5. Output Results
print(f"\n--- SEMANTIC MEANING RESULTS ---")
print(f"Total chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i+1} ---")
    print(chunk.page_content.strip())
