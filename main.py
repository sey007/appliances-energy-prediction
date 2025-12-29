from src.explore import run_exploration
from src.regression import run_regression

def main():
    print("=== Running EDA ===")
    run_exploration()

    print("\n=== Running Regression Model ===")
    run_regression()

if __name__ == "__main__":
    main()
