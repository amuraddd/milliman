"""
Just a helper file to trigger training
"""
import argparse
from models.xgboost_train_predict import search_best_params

if __name__=="__main__":
    #parse some argument passed through the command line
    parser = argparse.ArgumentParser(
        description="training arguments"
    )
    parser.add_argument(
        "--data_path",
        default="data",
        help="enable scaling of non-embedding featrues"
    )
    parser.add_argument(
        "--data_type",
        default="train",
        help="enable scaling of non-embedding featrues"
    )
    parser.add_argument(
        "--use_scalar",
        default=False,
        help="enable scaling of non-embedding featrues"
    )
    args = parser.parse_args()
    
    # launch param search
    search_best_params(
        data_path=args.data_path,
        data_type=args.data_type,
        use_scaler=args.use_scalar
    )