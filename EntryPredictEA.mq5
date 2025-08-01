#property strict
input double LotSize = 0.1;
input double RiskRewardRatio = 1.8;
input int MaxPositions = 13;         // 最大ポジション数
input double TakeProfitPips = 350;   // 利確幅（pips）
input double StopLossPips = 250;     // 損切り幅（pips）
input double EntryThreshold = 20;   // エントリー最低予測利益

int OnInit()
{
   Print("✅ EntryPredictEA initialized");
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if (PositionsTotal() >= MaxPositions) return;

   // ✅ features.json 出力
   double open  = iOpen(Symbol(), PERIOD_M1, 0);
   double high  = iHigh(Symbol(), PERIOD_M1, 0);
   double low   = iLow(Symbol(), PERIOD_M1, 0);
   double close = iClose(Symbol(), PERIOD_M1, 0);
   double ema200 = iMA(Symbol(), PERIOD_M1, 200, 0, MODE_EMA, PRICE_CLOSE);

   double body = MathAbs(close - open);
   double upper_wick = high - MathMax(open, close);
   double lower_wick = MathMin(open, close) - low;
   bool is_bull = close > open;
   bool is_bear = open > close;
   bool v_shape = false;

   string json = "{\n" +
                 "\"open\": " + DoubleToString(open, 2) + ",\n" +
                 "\"high\": " + DoubleToString(high, 2) + ",\n" +
                 "\"low\": " + DoubleToString(low, 2) + ",\n" +
                 "\"close\": " + DoubleToString(close, 2) + ",\n" +
                 "\"ema200\": " + DoubleToString(ema200, 2) + ",\n" +
                 "\"body\": " + DoubleToString(body, 2) + ",\n" +
                 "\"upper_wick\": " + DoubleToString(upper_wick, 2) + ",\n" +
                 "\"lower_wick\": " + DoubleToString(lower_wick, 2) + ",\n" +
                 "\"is_bull\": " + (is_bull ? "true" : "false") + ",\n" +
                 "\"is_bear\": " + (is_bear ? "true" : "false") + ",\n" +
                 "\"v_shape\": " + (v_shape ? "true" : "false") + "\n" +
                 "}";

   int handle = FileOpen("features.json", FILE_WRITE | FILE_ANSI);
   if (handle != INVALID_HANDLE)
   {
      FileWrite(handle, json);
      FileClose(handle);
      Print("📤 features.json を出力しました");
   }
   else
   {
      Print("❌ features.json の書き込みに失敗");
      return;
   }

   // ✅ result.txt 読み取り
   
   int result_handle = FileOpen("result.txt", FILE_READ | FILE_TXT);
   if (result_handle == INVALID_HANDLE)
   {
      Print("❌ result.txt の読み込みに失敗（ファイルが見つからないか読めない）");
      return;
   }

   string predicted_str = FileReadString(result_handle);
   FileClose(result_handle);

   Print("📃 読み込んだ文字列: ", predicted_str);
   double predicted_profit = StringToDouble(predicted_str);
   Print("📈 予測利益値: ", predicted_profit);

   if (predicted_profit < EntryThreshold)
   {
      Print("⛔ 予測利益が", EntryThreshold, "未満のため見送り");
      return;
   }

   double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double spread = ask - bid;
   Print("📐 スプレッド: ", spread);

   double minStopLevel = SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   Print("📏 最小ストップレベル: ", minStopLevel);

   // ✅ スプレッドと最小距離に基づいて安全距離を自動調整
   double safe_distance = MathMax(spread + minStopLevel, 5 * _Point);
   double tp_distance = TakeProfitPips * _Point + safe_distance;
   double sl_distance = StopLossPips * _Point + safe_distance;

   double stoploss = ask - sl_distance;
   double takeprofit = ask + tp_distance;

   Print("📈 エントリー価格: ", ask);
   Print("🛑 SL: ", stoploss, " / 🎯 TP: ", takeprofit);

   MqlTradeRequest request;
   MqlTradeResult tradeResult;
   ZeroMemory(request);
   ZeroMemory(tradeResult);

   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = Symbol();
   request.volume   = LotSize;
   request.type     = ORDER_TYPE_BUY;
   request.price    = ask;
   request.sl       = stoploss;
   request.tp       = takeprofit;
   request.deviation= 10;
   request.magic    = 123456;
   request.comment  = "Buy by ML (TP35 SL25 AutoSpread)";
   request.type_filling = ORDER_FILLING_IOC;

   if (!OrderSend(request, tradeResult))
      Print("❌ エントリー失敗: ", tradeResult.retcode, " / 詳細: ", tradeResult.comment);
   else
      Print("✅ エントリー成功: Ticket=", tradeResult.order);
}
