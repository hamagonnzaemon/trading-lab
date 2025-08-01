//+------------------------------------------------------------------+
//| EntryPredictEA.mq5 - ログ強化版                                 |
//+------------------------------------------------------------------+
#property strict
input double LotSize = 0.1;
input double RiskRewardRatio = 1.8;

int OnInit()
  {
   Print("✅ EntryPredictEA initialized");
   return(INIT_SUCCEEDED);
  }

void OnTick()
  {
   if (PositionsTotal() > 0) return;

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

   int handle = FileOpen("features.json", FILE_WRITE|FILE_TXT);
   if (handle != INVALID_HANDLE)
     {
      FileWrite(handle, json);
      FileClose(handle);
      Print("📤 features.json を出力しました");
     }
   else
     {
      Print("❌ JSONファイルの書き込みに失敗");
      return;
     }

   Sleep(3000);
   handle = FileOpen("result.txt", FILE_READ|FILE_TXT);
   if (handle != INVALID_HANDLE)
     {
      string predicted_str = FileReadString(handle);
      FileClose(handle);
      double predicted_profit = StringToDouble(predicted_str);

      Print("📥 result.txt 読み込み成功: ", predicted_str);
      Print("📈 予測利益値: ", predicted_profit);

      if (predicted_profit > 0)
        {
         double tp_distance = predicted_profit * _Point;
         double sl_distance = tp_distance / RiskRewardRatio;
         double price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
         double stoploss = price - sl_distance;
         double takeprofit = price + tp_distance;

         MqlTradeRequest request;
         MqlTradeResult tradeResult;
         ZeroMemory(request);
         ZeroMemory(tradeResult);

         request.action   = TRADE_ACTION_DEAL;
         request.symbol   = Symbol();
         request.volume   = LotSize;
         request.type     = ORDER_TYPE_BUY;
         request.price    = price;
         request.sl       = stoploss;
         request.tp       = takeprofit;
         request.deviation= 10;
         request.magic    = 123456;
         request.comment  = "Buy by ML";
         request.type_filling = ORDER_FILLING_IOC;

         if (!OrderSend(request, tradeResult))
            Print("❌ エントリー失敗: ", tradeResult.retcode);
         else
            Print("✅ エントリー成功: Ticket=", tradeResult.order);
        }
      else
        {
         Print("⛔ 予測利益がマイナス（", predicted_profit, "）のためエントリー見送り");
        }
     }
   else
     {
      Print("❌ result.txt の読み込みに失敗");
     }
  }