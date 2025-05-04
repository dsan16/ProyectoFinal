export interface Transaction {
    step: number;
    type: 'PAYMENT' | 'TRANSFER' | 'CASH_OUT' | 'DEBIT' | 'CASH_IN';
    amount: number;
    nameOrig: string;
    oldbalanceOrg: number;
    newbalanceOrig: number;
    nameDest: string;
    oldbalanceDest: number;
    newbalanceDest: number;
    isFraud: number;
    isFlaggedFraud: number;
  }
  