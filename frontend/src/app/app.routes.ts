import { Routes } from '@angular/router';
import { ShowAllComponent } from './components/transaction-list/show-all.component';
import { TransactionPredictComponent } from './components/prediction/prediction.component';

export const routes: Routes = [
    {path: '', redirectTo: 'transactions', pathMatch: 'full'},
    {path: 'transactions', component: ShowAllComponent},
    {path: 'prediction', component: TransactionPredictComponent},
];
