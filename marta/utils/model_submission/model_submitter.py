import pickle
from datetime import datetime
import os
import pandas as pd # type: ignore

class ModelSubmitter:
    def __init__(self, model:any, test_data:pd.DataFrame, feature_columns:list[str], index_column:str, filepath='Submission/'):
        self.model = model
        self.test_data = test_data
        self.feature_columns = feature_columns
        self.index_column = index_column
        self.filepath = filepath # where to save the model and submission file
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
    
    def get_predictions(self):
        X_test = self.test_data[self.feature_columns]
        predictions = self.model.predict(X_test)
        return pd.DataFrame({'carIDID': self.test_data[self.index_column], 'price': predictions})
    
    def save_submission(self, filename:str='submission.csv', submit_model:bool = False, submint_msg:str='Group 45 Model Submission'):
        predictions_df = self.get_predictions()
        # get today's date
        date_str = datetime.now().strftime('%Y%m%d')
        full_filename = f"{self.filepath}{date_str}_{filename}"
        count = 1
        while os.path.exists(full_filename):
            full_filename = f"{self.filepath}{date_str}_{count}_{filename}"
            count += 1
        predictions_df.to_csv(full_filename, index=False)
        print(f"Submission saved to {full_filename}")
        # now save the model
        model_filename = full_filename.replace('.csv', '_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_filename}")
        if submit_model:
            self.submit_trough_api(fullfilename=filename, filepath=self.filepath, message=submint_msg)
    @staticmethod
    def submit_trough_api(fullfilename:str , message:str ='Group 45 Model Submission'):
        """
        Submits the given submission file to Kaggle using the Kaggle API.

        to use this function, make sure you have the Kaggle API installed and configured.
        go to https://www.kaggle.com/docs/api for more information.
        Args:
            filename (str, optional): Defaults to 'submission.csv'.
            message (str, optional): Defaults to 'Group 45 Model Submission'.
            filepath (str, optional): Path to the submission file. Defaults to ''.

        Raises:
            FileNotFoundError: if the submission file does not exist.
        """
        
        if not os.path.exists(f"{fullfilename}"):
            raise FileNotFoundError(f"Submission file {fullfilename} not found. Please save the submission first.")
        
        system_command = f'kaggle competitions submit -c cars4you -f {fullfilename} -m "{message}"'
        ret = os.system(system_command)
        if ret == 0:
            print(f"Submission {fullfilename} successfully sent to Kaggle.")
        else:
            print(f"Failed to submit {fullfilename} to Kaggle with code {ret}. Please check the Kaggle API configuration.")