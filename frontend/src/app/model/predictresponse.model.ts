export interface PredictResponse {
    probabilidad_fraude: number;
    es_fraude: number;
    orig_tx_count: number;
    dest_tx_count: number;
}  