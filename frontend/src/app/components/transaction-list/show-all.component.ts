import { Component, OnInit }       from '@angular/core';
import { CommonModule }            from '@angular/common';
import { FormsModule }          from '@angular/forms'; 
import { Transaction }             from '../../model/transactions.model';
import { TransactionService }      from '../../services/transaction.service';
import { NgForOf, SlicePipe }      from '@angular/common';
import { RouterModule }         from '@angular/router';  

@Component({
  selector: 'app-show-all',
  standalone: true,
  imports: [
    CommonModule,
    NgForOf,
    SlicePipe,
    FormsModule,
    RouterModule
  ],
  templateUrl: './show-all.component.html',
  styleUrls: ['./show-all.component.css']
})
export class ShowAllComponent implements OnInit {
  transactions: Transaction[] = [];
  pageIndex = 0;
  pageSize = 10;

  filterName: string = '';
  filterLimit: number = 10;

  constructor(private myService: TransactionService) {}

  get totalPages(): number {
    return Math.ceil(this.transactions.length / this.pageSize);
  }

  prevPage(): void {
    if (this.pageIndex > 0) this.pageIndex--;
  }

  nextPage(): void {
    if (this.pageIndex + 1 < this.totalPages) this.pageIndex++;
  }

  ngOnInit(): void {
    this.loadAll();
  }

  /** Carga todas las transacciones */
  loadAll(): void {
    this.myService.getAllTransactions().subscribe(list => {
      this.transactions = list;
      this.pageIndex = 0;
    });
  }

  /** Carga solo las transacciones de fraude */
  loadFrauds(): void {
    this.myService.getFraudTransactions().subscribe(list => {
      this.transactions = list;
      this.pageIndex = 0;
    });
  }

  loadByOrigin(): void {
    if (!this.filterName) { return; }
    this.myService.getTransactionsByOrigin(this.filterName, this.filterLimit)
      .subscribe((list: Transaction[]) => {
        this.transactions = list;
        this.pageIndex = 0;
      });
  }
}
