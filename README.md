
```
trading_system
├─ .vscode
│  └─ settings.json
├─ data
│  ├─ backtesting
│  ├─ backtest_results.csv
│  ├─ config.ini
│  ├─ models
│  │  ├─ model_catboost.joblib
│  │  ├─ model_cb_two_scalars_multiplefeatures.joblib
│  │  ├─ model_rf.joblib
│  │  ├─ model_rf_results.txt
│  │  ├─ model_rf_two_scalars.joblib
│  │  ├─ model_rf_two_scalars_multiplefeatures_15_feature_selection.joblib
│  │  ├─ model_rf_two_scalars_multiplefeatures_15_feature_selection_32_estimators.joblib
│  │  ├─ model_rf_two_scalars_multiplefeatures_25_feature_selection_32_estimators.joblib
│  │  └─ test.joblib
│  ├─ predictions
│  │  ├─ excel_output
│  │  │  ├─ 2024-06-26.xlsx
│  │  │  ├─ 2024-06-27.xlsx
│  │  │  ├─ 2024-06-28.xlsx
│  │  │  ├─ 2024-07-01.xlsx
│  │  │  ├─ 2024-07-02.xlsx
│  │  │  ├─ 2024-07-03.xlsx
│  │  │  ├─ 2024-07-04.xlsx
│  │  │  └─ 2024-07-05.xlsx
│  │  ├─ predictions.pkl
│  │  ├─ predictions.xlsx
│  │  └─ raw_predictions.pkl
│  ├─ processed
│  │  ├─ UpstoxData.pkl
│  │  └─ UpstoxDataWithIndicators.pkl
│  └─ raw
│     └─ UpstoxData.db
├─ Main
│  └─ Logs
│     └─ Data Downloader
│        └─ 20240624_230210_data_download.log
├─ notebooks
│  ├─ catboost_info
│  │  ├─ catboost_training.json
│  │  ├─ learn
│  │  │  └─ events.out.tfevents
│  │  ├─ learn_error.tsv
│  │  ├─ time_left.tsv
│  │  └─ tmp
│  │     ├─ cat_feature_index.2ba60db5-1476a012-b9377699-27b80161.tmp
│  │     └─ cat_feature_index.57b02698-add389c6-7c377861-5af4346a.tmp
│  ├─ check_data.ipynb
│  └─ research_strategy.ipynb
└─ scripts
   ├─ add_indicators.py
   ├─ add_ind_train_save_model.py
   ├─ Archive
   │  ├─ signals.py
   │  ├─ signals_v2.py
   │  └─ signals_v3.py
   ├─ backtest.py
   ├─ fetch_data.py
   ├─ portfolio_size.py
   ├─ process_data.py
   └─ rfe.py

```
```
trading_system
├─ .vscode
│  └─ settings.json
├─ catboost_info
│  ├─ catboost_training.json
│  ├─ learn
│  │  └─ events.out.tfevents
│  ├─ learn_error.tsv
│  ├─ time_left.tsv
│  └─ tmp
│     ├─ cat_feature_index.ca6a5681-3141e1c3-c5564a9a-eb5d6e74.tmp
│     └─ cat_feature_index.ddd71dff-94b2c1b2-4fa98a8f-a0e2c082.tmp
├─ data
│  ├─ archive
│  │  └─ backtest_results_till_09_07_24.csv
│  ├─ backtesting
│  ├─ backtest_results.csv
│  ├─ config.ini
│  ├─ keys
│  │  └─ tradingsystem-428613-9a2f6e64d8a5.json
│  ├─ models
│  │  ├─ model_catboost.joblib
│  │  ├─ model_cb_two_scalars_multiplefeatures.joblib
│  │  ├─ model_rf.joblib
│  │  ├─ model_rf_results.txt
│  │  ├─ model_rf_two_scalars.joblib
│  │  ├─ model_rf_two_scalars_multiplefeatures_15_feature_selection.joblib
│  │  ├─ model_rf_two_scalars_multiplefeatures_15_feature_selection_32_estimators.joblib
│  │  ├─ model_rf_two_scalars_multiplefeatures_25_feature_selection_32_estimators.joblib
│  │  ├─ random_forest_20240706-181426_test.joblib
│  │  └─ test.joblib
│  ├─ predictions
│  │  ├─ excel_output
│  │  │  ├─ 2024-06-26.xlsx
│  │  │  ├─ 2024-06-27.xlsx
│  │  │  ├─ 2024-06-28.xlsx
│  │  │  ├─ 2024-07-01.xlsx
│  │  │  ├─ 2024-07-02.xlsx
│  │  │  ├─ 2024-07-03.xlsx
│  │  │  ├─ 2024-07-04.xlsx
│  │  │  ├─ 2024-07-05.xlsx
│  │  │  └─ 2024-07-08.xlsx
│  │  ├─ predictions.pkl
│  │  ├─ predictions.xlsx
│  │  └─ raw_predictions.pkl
│  ├─ processed
│  │  ├─ UpstoxData.pkl
│  │  └─ UpstoxDataWithIndicators.pkl
│  └─ raw
│     └─ UpstoxData.db
├─ Main
│  └─ Logs
│     └─ Data Downloader
│        └─ 20240624_230210_data_download.log
├─ notebooks
│  ├─ catboost_info
│  │  ├─ catboost_training.json
│  │  ├─ learn
│  │  │  └─ events.out.tfevents
│  │  ├─ learn_error.tsv
│  │  ├─ time_left.tsv
│  │  └─ tmp
│  │     ├─ cat_feature_index.2ba60db5-1476a012-b9377699-27b80161.tmp
│  │     └─ cat_feature_index.57b02698-add389c6-7c377861-5af4346a.tmp
│  ├─ check_data.ipynb
│  └─ research_strategy.ipynb
├─ README.md
└─ scripts
   ├─ add_indicators.py
   ├─ add_ind_train_save_model.py
   ├─ Archive
   │  ├─ signals.py
   │  ├─ signals_v2.py
   │  └─ signals_v3.py
   ├─ backtest.py
   ├─ fetch_data.py
   ├─ generate_signals.py
   ├─ make_predictions.py
   ├─ portfolio_size.py
   ├─ process_data.py
   ├─ rfe.py
   └─ __pycache__
      ├─ add_indicators.cpython-310.pyc
      ├─ fetch_data.cpython-310.pyc
      ├─ generate_signals.cpython-310.pyc
      ├─ make_predictions.cpython-310.pyc
      └─ process_data.cpython-310.pyc

```