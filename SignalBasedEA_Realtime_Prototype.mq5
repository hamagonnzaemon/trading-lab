//+------------------------------------------------------------------+
//|  SignalBasedEA_Realtime_Prototype.mq5                            |
//|  STEP-1 : CSV 出力（120 s）＋バー確定時                            |
//|  STEP-3 : signal_to_ea.txt を毎 tick 監視                          |
//|  Version: 5.09  (2025-06-10 FileMove flag fix)                   |
//+------------------------------------------------------------------+
#property copyright "OpenAI & User"
#property version   "5.09"
#property strict
#include <Trade/Trade.mqh>

//====== ① パラメータ =================================================
input group "Basic Settings"
input int      InpToleranceSec = 60;
input double   InpLots         = 0.10;
input int      InpMagicNumber  = 12345;
input int      InpSlippagePoints = 20;

input group "Fixed TP/SL Settings"
input double   InpTP_pips = 30.0;
input double   InpSL_pips = 20.0;

input group "Python Link (STEP-1: Data Output)"
input string   InpPythonBarsFile             = "realtime_bars_for_python.csv";
input int      InpPythonDataOutputIntervalSec = 120;   // ← 120 秒
input int      InpNumBarsToOutput             =300;
input int      InpATR_Period_ForPython       = 14;
input int      InpK_Period_ForPython         = 15;
input int      InpD_Period_ForPython         = 9;
input int      InpSlowing_ForPython          = 3;
input int      InpEMAFast_Period_ForPython   = 20;
input int      InpEMAMid_Period_ForPython    = 50;
input int      InpEMASlow_Period_ForPython   = 70;

input group "Python Link (STEP-3: Signal Input)"
input string   InpPythonSignalFile           = "signal_to_ea.txt"; // ★このファイルは MQL5\Files フォルダに配置

//====== ② グローバル変数・ハンドル ====================================
CTrade trade;

int gi_ATR_handle_python,
    gi_Stoch_handle_python,
    gi_EMA_Fast_handle_python,
    gi_EMA_Mid_handle_python,
    gi_EMA_Slow_handle_python;

datetime g_last_signal_file_mod_time_checked = 0; // 前回シグナルファイルを確認した時刻
string   g_last_read_signal_content          = "";  // 前回読み込んだシグナルの内容
datetime g_last_bar_written                  = 0;

const int SHIFT = 1;       // 確定バー (共通)

//====== ③ OnInit ======================================================
int OnInit()
{
   Print("[EA] v5.09 通知機能 Initializing…");

   gi_ATR_handle_python    = iATR(_Symbol,Period(),InpATR_Period_ForPython);
   gi_Stoch_handle_python = iStochastic(_Symbol,Period(),
                                       InpK_Period_ForPython,
                                       InpD_Period_ForPython,
                                       InpSlowing_ForPython,
                                       MODE_SMA,
                                       STO_LOWHIGH);       // HLC 計算

   gi_EMA_Fast_handle_python = iMA(_Symbol,Period(),InpEMAFast_Period_ForPython,0,MODE_EMA,PRICE_CLOSE);
   gi_EMA_Mid_handle_python  = iMA(_Symbol,Period(),InpEMAMid_Period_ForPython ,0,MODE_EMA,PRICE_CLOSE);
   gi_EMA_Slow_handle_python = iMA(_Symbol,Period(),InpEMASlow_Period_ForPython,0,MODE_EMA,PRICE_CLOSE);

   if(InpPythonDataOutputIntervalSec > 0)
     EventSetTimer(InpPythonDataOutputIntervalSec);

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippagePoints);
   return(INIT_SUCCEEDED);
}

//====== ④ OnDeinit ====================================================
void OnDeinit(const int r)
{
   if(InpPythonDataOutputIntervalSec > 0) EventKillTimer();
   if(gi_ATR_handle_python    != INVALID_HANDLE) IndicatorRelease(gi_ATR_handle_python);
   if(gi_Stoch_handle_python != INVALID_HANDLE) IndicatorRelease(gi_Stoch_handle_python);
   if(gi_EMA_Fast_handle_python!= INVALID_HANDLE) IndicatorRelease(gi_EMA_Fast_handle_python);
   if(gi_EMA_Mid_handle_python != INVALID_HANDLE) IndicatorRelease(gi_EMA_Mid_handle_python);
   if(gi_EMA_Slow_handle_python!= INVALID_HANDLE) IndicatorRelease(gi_EMA_Slow_handle_python);
}

//====== ⑤ OnTimer : 120 s ごとのループ ================================
void OnTimer()
{
   OutputDataForPython();         // CSV 書き出し
   CheckAndProcessPythonSignal();  // シグナル確認
}

//====== ⑤’ OnTick : バー確定直後 / 毎 tick =============================
void OnTick()
{
   // 5 分バー確定判定
   datetime last_closed = iTime(_Symbol, PERIOD_M5, SHIFT);
   if(last_closed != g_last_bar_written && last_closed > 0)
   {
     g_last_bar_written = last_closed;
     OutputDataForPython();     // 確定バー分を即出力
   }

   CheckAndProcessPythonSignal(); // ファイル更新が速い場合に備え tick でも見る
}

//====== ⑥ OutputDataForPython (shift = 1) =============================
void OutputDataForPython()
{
   int need = InpNumBarsToOutput;
   if(Bars(_Symbol,Period()) < need + SHIFT) return;

   MqlRates rates[];         ArraySetAsSeries(rates,true);
   double atr[],kbuf[],dbuf[],emaF[],emaM[],emaS[];
   ArraySetAsSeries(atr,true); ArraySetAsSeries(kbuf,true); ArraySetAsSeries(dbuf,true);
   ArraySetAsSeries(emaF,true);ArraySetAsSeries(emaM,true);ArraySetAsSeries(emaS,true);

   if(CopyRates (_Symbol,Period(),SHIFT,need,rates)<need) return;
   if(CopyBuffer(gi_ATR_handle_python   ,0,SHIFT,need,atr )<need) return;
   if(CopyBuffer(gi_Stoch_handle_python,0,SHIFT,need,kbuf)<need) return;
   if(CopyBuffer(gi_Stoch_handle_python,1,SHIFT,need,dbuf)<need) return;
   if(CopyBuffer(gi_EMA_Fast_handle_python,0,SHIFT,need,emaF)<need) return;
   if(CopyBuffer(gi_EMA_Mid_handle_python ,0,SHIFT,need,emaM)<need) return;
   if(CopyBuffer(gi_EMA_Slow_handle_python,0,SHIFT,need,emaS)<need) return;

   // CSVファイルは MQL5\Files フォルダに出力されます
   string tmp_file_path = InpPythonBarsFile + ".tmp";
   // FileOpenの第4引数は区切り文字、第5引数 common_folder_flag (falseでMQL5\Files)
   int fh = FileOpen(tmp_file_path, FILE_WRITE|FILE_CSV|FILE_ANSI, ',', false);
   if(fh == INVALID_HANDLE){ PrintFormat("[EA] CSV temp file open error for '%s'. Error: %d", tmp_file_path, GetLastError()); return; }

   FileWriteString(fh,"bar_time,open,high,low,close,tick_volume,"
                     "ATR_14,K,D,ema_fast,ema_mid,ema_slow,KD_angle\n");

   int dp = (int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   for(int i = need-1; i >= 0; i--)
   {
     double kd_norm = (kbuf[i]-dbuf[i])/100.0;       // 0-1
     double kd_deg  = MathArctan(kd_norm)*180.0/M_PI;

     FileWrite(fh,
       TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
       rates[i].open, rates[i].high, rates[i].low, rates[i].close,
       rates[i].tick_volume,
       atr[i], kbuf[i], dbuf[i],
       emaF[i], emaM[i], emaS[i],
       kd_deg);
   }
   FileClose(fh);

   // MQL5\Files 内のファイルを操作
   string target_file_path = InpPythonBarsFile;
   // FileIsExistの第2引数 common_folder_flag (falseでMQL5\Files)
   if(FileIsExist(target_file_path, false))
   {
      // FileDeleteの第2引数 common_folder_flag (falseでMQL5\Files)
      FileDelete(target_file_path, false);
   }
   // FileMove(ソースファイル名, ソースのcommon_flag, 保存先ファイル名, 上書きフラグ)
   // ソースのcommon_flag: 0 または false で MQL5\Files
   // 保存先ファイル名: 相対パスなら MQL5\Files へ
   // 上書きフラグ: FILE_REWRITE で上書き
   if(!FileMove(tmp_file_path, 0, target_file_path, FILE_REWRITE)) // ★修正点: 不明な識別子 FILE_COMMON_FOLDER_FALSE を削除
   {
      PrintFormat("[EA] Failed to move '%s' to '%s'. Error: %d", tmp_file_path, target_file_path, GetLastError());
   } else {
      PrintFormat("[EA] CSV '%s' output complete.", target_file_path);
   }
}

//====== ⑦ シグナル処理 (★改修版) ===============================================
void CheckAndProcessPythonSignal()
{
    // 条件1: 既に何らかのポジションを保有している場合は、新しいシグナルを処理しない
    if(PositionSelect(_Symbol)) {
        // PrintFormat("[EA] Position exists for %s. Skipping signal check.", _Symbol); // デバッグ用
        return;
    }

    string signal_file_path = InpPythonSignalFile;

    // 条件2: シグナルファイルが MQL5\Files (またはテスト時は Tester\Files) に存在するか確認
    // FileIsExistの第2引数に false を指定すると、MQL5\Files を見ます。(省略時も同様)
    if(!FileIsExist(signal_file_path, false)) {
        // PrintFormat("[EA] Signal file '%s' not found in MQL5/Files.", signal_file_path); // デバッグ用
        return;
    }

    // シグナルファイルの最終更新日時を取得 (MQL5\Files を見る)
    // FileGetIntegerの第3引数に false を指定
    long mod_long = FileGetInteger(signal_file_path, FILE_MODIFY_DATE, false);
    if(mod_long == 0) { // 更新日時が取得できなかった場合 (ファイルが存在しない、アクセス権がないなど)
        PrintFormat("[EA] Failed to get modification date for signal file '%s' (in MQL5/Files). Error: %d", signal_file_path, GetLastError());
        return;
    }
    datetime current_mod_time = (datetime)mod_long;

    // シグナルファイルを開く (MQL5\Files から)
    // FileOpenの第4引数に false を指定 (または省略)
    // FILE_SHARE_READ を追加して、Pythonが書き込み中でも読み取りエラーになりにくくする
    int fh = FileOpen(signal_file_path, FILE_READ|FILE_ANSI|FILE_SHARE_READ, ',', false);
    if(fh == INVALID_HANDLE) {
        PrintFormat("[EA] Error opening signal file '%s' (in MQL5/Files). Error: %d", signal_file_path, GetLastError());
        return;
    }

    // ファイル内容を読み込む
    string current_signal_content = "";
    if(!FileIsEnding(fh)) { // ファイルが空でないことを確認
        current_signal_content = FileReadString(fh);
        StringTrimLeft(current_signal_content);
        StringTrimRight(current_signal_content);
    }
    FileClose(fh);

    // --- デバッグ用ログ (必要に応じてコメントを解除してください) ---
    /*
    PrintFormat("[EA] Signal Check: File='%s'\n"
                "  Current ModTime: %s (LastChecked: %s)\n"
                "  Current Content: '%s' (LastRead: '%s')",
                signal_file_path,
                TimeToString(current_mod_time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
                TimeToString(g_last_signal_file_mod_time_checked, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
                current_signal_content,
                g_last_read_signal_content);
    */
    // --- デバッグ用ログここまで ---

    // (A) ファイル内容が空の場合の処理
    if(current_signal_content == "") {
        // ファイルが空になった、または以前から空でファイルだけ更新された場合
        if (g_last_read_signal_content != "" || (g_last_read_signal_content == "" && current_mod_time > g_last_signal_file_mod_time_checked)) {
            PrintFormat("[EA] Signal file '%s' is now empty or cleared. Last non-empty content was '%s'.", signal_file_path, g_last_read_signal_content != "" ? g_last_read_signal_content : "N/A");
            g_last_signal_file_mod_time_checked = current_mod_time; // 最終確認日時を更新
            g_last_read_signal_content = "";                       // 空の内容を保存
        }
        return; // 空なので発注処理はしない
    }

    // (B) 新しいシグナルを処理するかどうかの判定 (ファイル内容が空でない場合)
    bool should_process_signal = false;

    if (g_last_signal_file_mod_time_checked == 0) {
        // 初回読み込み時 (かつファイル内容が空でない)
        should_process_signal = true;
        PrintFormat("[EA] First time reading. Processing signal: '%s'", current_signal_content);
    } else if (g_last_read_signal_content != current_signal_content) {
        // 前回読み込んだ内容と現在の内容が異なる場合 (例: "BUY" -> "SELL")
        should_process_signal = true;
        PrintFormat("[EA] Signal content changed from '%s' to '%s'. Processing.", g_last_read_signal_content, current_signal_content);
    } else if (current_mod_time > g_last_signal_file_mod_time_checked) {
        // 内容は前回と同じだが、ファイルが更新された場合 (Pythonが同じシグナルを上書きした等)
        // この場合は新しい発注は行わず、最終確認日時のみ更新する
        PrintFormat("[EA] Signal content '%s' is unchanged, but file timestamp updated. No new order. Updating check time.", current_signal_content);
        g_last_signal_file_mod_time_checked = current_mod_time;
        // g_last_read_signal_content は同じなので更新不要
        return; // 発注ロジックには進まない
    }
    // 上記以外 (更新日時も内容も前回と同じ) の場合は何もしない

    if(should_process_signal) {
        PrintFormat("[EA] Processing signal command: '%s' (File mtime: %s)", current_signal_content, TimeToString(current_mod_time,TIME_DATE|TIME_MINUTES|TIME_SECONDS));

        if(current_signal_content == "BUY") {
            OpenBuyOrderFixed();
        }
        // else if(current_signal_content == "SELL") {
        //     OpenSellOrderFixed(); // SELL注文用の関数を別途実装する必要があります
        // }
        // 他のシグナルコマンドにも対応する場合はここに追加

        // 処理後、最終情報を更新
        g_last_signal_file_mod_time_checked = current_mod_time;
        g_last_read_signal_content = current_signal_content;
    }
}


//====== ⑧ 発注 ========================================================
bool OpenBuyOrderFixed()
{
   MqlTick tk; if(!SymbolInfoTick(_Symbol,tk)) return false;
   double ask = tk.ask; if(ask <= 0) return false;

   double tp = ask + GetPriceOffsetFromPips(InpTP_pips);
   double sl = ask - GetPriceOffsetFromPips(InpSL_pips);

   int    dg   = (int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   double pt   = SymbolInfoDouble (_Symbol,SYMBOL_POINT);
   long   stop = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   double tick_size = SymbolInfoDouble (_Symbol,SYMBOL_TRADE_TICK_SIZE);

   if(InpTP_pips > 0 && tp <= NormalizeDouble(ask + stop * pt, dg)) {
       tp = NormalizeDouble(ask + stop * pt + tick_size, dg);
       PrintFormat("[EA] TP for BUY adjusted to %s due to stop level %d points.", DoubleToString(tp, dg), stop);
   }
   if(InpSL_pips > 0 && sl >= NormalizeDouble(ask - stop * pt, dg)) {
       sl = NormalizeDouble(ask - stop * pt - tick_size, dg);
       PrintFormat("[EA] SL for BUY adjusted to %s due to stop level %d points.", DoubleToString(sl, dg), stop);
   }
   if (sl <=0) sl = 0.0;

   trade.SetDeviationInPoints(InpSlippagePoints);
   if(trade.Buy(InpLots,_Symbol,ask,sl,tp,"PySigBuy"))
   {
      // ▼▼▼ ここを修正 ▼▼▼
      string msg = StringFormat("%c [EA] BUY order success: %.2f lot @ %.5f TP=%.5f SL=%.5f", 10005, InpLots, ask, tp, sl);
      Print(msg);
      SendNotification(msg);  // スマホ通知 (MT5設定必須)
      Alert(msg);             // PC音 + ポップアップ
      return true;
   }

    // ▼▼▼ ここを修正 ▼▼▼
    string err = StringFormat("%c [EA] BUY order failed. Ret=%d %s", 10060, trade.ResultRetcode(), trade.ResultRetcodeDescription());
    Print(err);
    SendNotification(err);
    Alert(err);
    return false;
}

//====== ⑨ pips → PriceOffset ========================================
// 指定されたpips数から価格差を計算する
double GetPriceOffsetFromPips(double pips_value)
{
    // SYMBOL_POINT: 1ポイントの価格単位 (例: EURUSDなら0.00001, USDJPYなら0.001)
    double point_value = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    // SYMBOL_DIGITS: 価格の小数点以下の桁数
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // JPYペアやGOLDなど、"1 pip" の定義が通常と異なる場合があるため、桁数で調整
    // 一般的なFXペア (5桁/3桁表示) の場合、1 pip = 10 points
    // 例: EURUSD (1.23456) -> 1 pip = 0.00010
    // 例: USDJPY (123.456) -> 1 pip = 0.010
    double pips_multiplier = 1.0;
    if (digits == 5 || digits == 3) { // 5桁(EURUSD等)または3桁(USDJPY等)表示の場合
        pips_multiplier = 10.0;
    } else if (digits == 4 || digits == 2) { // 稀なケース (4桁/2桁表示)
        // この場合は 1 pip = 1 point となるが、通常は上記でカバーされる
         pips_multiplier = 1.0;
    }
    // GOLD(XAUUSD)などは特殊な場合があるので、必要に応じて個別対応
    // string current_symbol = Symbol();
    // if (StringFind(current_symbol, "XAU") != -1 || StringFind(current_symbol, "GOLD") != -1) {
    //    // GOLDの場合のpips計算ロジック (例: 1 pip = 0.1 USD)
    // }

    return NormalizeDouble(pips_value * pips_multiplier * point_value, digits);
}
//+------------------------------------------------------------------+
