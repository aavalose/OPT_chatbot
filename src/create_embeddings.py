from embeddings_manager import EmbeddingsManager

def main():
    """Create embeddings and vector store for all questions"""
    manager = EmbeddingsManager()
    manager.process_json_files()

if __name__ == "__main__":
    main() 