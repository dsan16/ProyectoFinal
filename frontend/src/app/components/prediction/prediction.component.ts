import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { TransactionService } from '../../services/transaction.service';
import { Transaction } from '../../model/transactions.model';
import { PredictResponse } from '../../model/predictresponse.model';
import { RouterModule }         from '@angular/router';  

@Component({
  selector: 'app-transaction-predict',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterModule],
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css']
})
export class TransactionPredictComponent {
  form: FormGroup;
  result?: PredictResponse;

  tipos: Transaction['type'][] = ['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'];

  constructor(
    private fb: FormBuilder,
    private service: TransactionService
  ) {
    this.form = this.fb.group({
      type:        ['PAYMENT', Validators.required],
      amount:      [0, Validators.required],
      nameOrig:    ['', Validators.required],
      oldbalanceOrg:   [0, Validators.required],
      newbalanceOrig:  [0, Validators.required],
      nameDest:    ['', Validators.required],
      oldbalanceDest:  [0, Validators.required],
      newbalanceDest:  [0, Validators.required]
    });
  }

  submit(): void {
    if (this.form.invalid) return;
    const payload = this.form.value as Omit<Transaction, 'step'|'isFraud'|'isFlaggedFraud'>;
    this.service.predictTransaction(payload)
      .subscribe(res => this.result = res);
  }

  insert(): void {
    if (this.form.invalid) return;
    const base = this.form.value as Omit<Transaction, 'step'|'isFraud'|'isFlaggedFraud'>;

    this.service.predictTransaction(base).subscribe(res => {
      const isFraudValue = res.es_fraude;

      const toInsert: Transaction = {
        step: 0,
        type: base.type,
        amount: base.amount,
        nameOrig: base.nameOrig,
        oldbalanceOrg: base.oldbalanceOrg,
        newbalanceOrig: base.newbalanceOrig,
        nameDest: base.nameDest,
        oldbalanceDest: base.oldbalanceDest,
        newbalanceDest: base.newbalanceDest,
        isFraud: isFraudValue,
        isFlaggedFraud: 0
      };

      this.service.insertTransaction(toInsert)
        .subscribe(() => {
          alert('Transacción insertada correctamente');
        });
    });
  }
}
