from src.data.data_preprocessing import load_and_preprocess_data
from src.models.model import train_and_evaluate_model

def main():
    load_and_preprocess_data()
    train_and_evaluate_model()

if __name__ == "__main__":
    main()
