from src.preprocessing import load_data, preprocess_data
from src.models import train_models
from src.evaluation import evaluate_models, create_report

def main():
    # Load and preprocess data
    data = load_data("data/cancer_dataset.csv")
    processed_data = preprocess_data(data)
    
    # Train models
    models = train_models(processed_data)
    
    # Evaluate models
    evaluation_results = evaluate_models(models, processed_data)
    
    # Create report
    create_report(evaluation_results)

if __name__ == "__main__":
    main()
#print("Hello World")