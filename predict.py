"""
Helper script to predict from a trained model
"""
import argparse
from models.xgboost_train_predict import get_model_predictions

if __name__=="__main__":
    #parse some argument passed through the command line
    parser = argparse.ArgumentParser(
        description="training arguments"
    )
    parser.add_argument(
        "--model_path",
        default="models/saved_models/xgboost_model.pkl",
        help="path of the save model"
    )
    parser.add_argument(
        "--save_predictions_path",
        default="data/predictions/",
        help="make predictions and save in a specified location"
    )
    args = parser.parse_args()
    
    # make predictions from the trained model
    # location of validation data is a default in the function but can be passed in and run time
    get_model_predictions(
        model_path=args.model_path,
        save_predictions_path=args.save_predictions_path
    )
    