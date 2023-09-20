import pandas as pd
import pickle
from category_encoders import TargetEncoder


def load_model(name):
    arq = 'models/' + name + '.pkl'
    with open(arq, 'rb') as file:
        model = pickle.load(file)
    print('Model '+ name + ' loaded')
    return model


class Classifier:
    def __init__(self):
        self.ada_classifier = load_model('ada_classifier')
        self.scaler = load_model('std_scaler')
        self.target_encoder = load_model('target_encoder')

    def prepare_data(self, data: list) -> pd.DataFrame:
        input_data = pd.DataFrame([data], columns=['customer_gender', 'customer_SeniorCitizen',
                                                    'customer_Partner', 'customer_Dependents', 'customer_tenure',
                                                    'phone_PhoneService', 'phone_MultipleLines',
                                                    'internet_InternetService',
                                                    'internet_OnlineSecurity', 'internet_OnlineBackup',
                                                    'internet_DeviceProtection', 'internet_TechSupport',
                                                    'internet_StreamingTV', 'internet_StreamingMovies', 'account_Contract',
                                                    'account_PaperlessBilling', 'account_PaymentMethod',
                                                    'account_Charges_Monthly', 'account_Charges_Total'])
        return input_data

    def treat_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        input_data = self.target_encoder.transform(input_data)
        input_data = self.scaler.transform(input_data)

        return input_data

    def predict(self, input_data: list) -> int:
        input_data = self.prepare_data(input_data)
        input_data = self.treat_data(input_data)

        return self.ada_classifier.predict(input_data), self.ada_classifier.predict_proba(input_data)






