from ddGPredictor.predictor import Predictor

def main():
    """
    Demonstrates how to use the Predictor class.
    """
    try:
        # Initialize predictor
        predictor = Predictor()

        # Example mutation
        mutation = {"AA_wt": "A", "AA_mut": "G"}

        # Predict
        prediction = predictor.predict(mutation)
        print(f"Mutation: {mutation}")
        print(f"Predicted ddG sign: {'Positive' if prediction == 1 else 'Negative or Zero'}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run compare_models.py to train and save the model first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()