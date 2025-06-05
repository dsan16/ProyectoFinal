import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Transaction } from '../model/transactions.model';
import { Observable } from 'rxjs';
import { PredictResponse } from '../model/predictresponse.model';

@Injectable({
  providedIn: 'root'
})
export class TransactionService {
  private readonly API_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  // Devuelve todas las transacciones 
  getAllTransactions(): Observable<Transaction[]> {
    return this.http.get<Transaction[]>(`${this.API_URL}/transactions/all`);
  }

  getFraudTransactions(): Observable<Transaction[]> {
    return this.http.get<Transaction[]>(`${this.API_URL}/transactions/allFraud`);
  }

  // Devuelve un muestreo de transacciones
  getSampleTransactions(limit: number = 10): Observable<Transaction[]> {
    return this.http.get<Transaction[]>(`${this.API_URL}/transactions/sample?limit=${limit}`);
  }

  // Devuelve transacciones filtradas por nameOrig 
  getTransactionsByOrigin(nameOrig: string, limit: number = 10): Observable<Transaction[]> {
    const params = { nameOrig, limit: limit.toString() };
    return this.http.get<Transaction[]>(`${this.API_URL}/transactions/by_origin`, { params });
  }

  // Predice si una transacción es fraudulenta o no y devuelve la probabilidad de fraude
  predictTransaction(
    transaction: Omit<Transaction, 'step' | 'isFraud' | 'isFlaggedFraud'>
  ): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(
      `${this.API_URL}/predict`,
      transaction
    );
  }

  insertTransaction(payload: Omit<Transaction, 'step'|'isFraud'|'isFlaggedFraud'>): Observable<any> {
    return this.http.post(`${this.API_URL}/transactions/insert`, payload);
  }
} 
