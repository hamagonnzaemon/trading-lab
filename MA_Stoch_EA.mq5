//+------------------------------------------------------------------+
//| MA_Stoch_EA v2.01 – long-only ENTRY logging only                |
//+------------------------------------------------------------------+
#property version   "2.01"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

CTrade  trade;
int     fh = INVALID_HANDLE;
bool    armed = false;
double  prevK = EMPTY_VALUE, prevD = EMPTY_VALUE;
int     hEmaFast, hEmaMid, hEmaSlow, hStoch;

//+------------------------------------------------------------------+
//| 初期化                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
  // Common\Files フォルダ直下に CSV を作成／追記
  fh = FileOpen("trade_log.csv",
                FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,
                ';');
  if(fh == INVALID_HANDLE)
  {
    Print("⚠ CSVオープン失敗 err=", GetLastError());
    return(INIT_FAILED);
  }
  // 新規ファイルならヘッダー行を書き込む
  if(FileSize(fh) == 0)
  {
    FileWrite(fh,
      "datetime","ticket",
      "K","D","crossStrength",
      "emaFast","emaMid","emaSlow","emaGap",
      "ask","price_vs_emaFast"
    );
  }
  FileSeek(fh, 0, SEEK_END);

  // インジハンドル作成
  hEmaFast = iMA(_Symbol, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE);
  hEmaMid  = iMA(_Symbol, PERIOD_M5, 50, 0, MODE_EMA, PRICE_CLOSE);
  hEmaSlow = iMA(_Symbol, PERIOD_M5, 70, 0, MODE_EMA, PRICE_CLOSE);
  hStoch   = iStochastic(_Symbol, PERIOD_M5, 15, 3, 9, MODE_SMA, STO_LOWHIGH);

  // prevK/prevD 初期化
  double k0[1], d0[1];
  if(CopyBuffer(hStoch,0,0,1,k0)==1 && CopyBuffer(hStoch,1,0,1,d0)==1)
  {
    prevK = k0[0];
    prevD = d0[0];
  }
  return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| ティック処理                                                     |
//+------------------------------------------------------------------+
void OnTick()
{
  // 1) ストキャス取得
  double kArr[1], dArr[1];
  if(CopyBuffer(hStoch,0,0,1,kArr)!=1 || CopyBuffer(hStoch,1,0,1,dArr)!=1) 
    return;
  double K = kArr[0], D = dArr[0];

  // 2) armed 制御 (80触れ→armed / 20触れ→解除)
  if(K >= 80.0) armed = true;
  if(K <= 20.0) armed = false;

  // 3) ゴールデンクロス検出
  bool  crossedUp     = (prevK < prevD && K >= D);
  double crossStrength = K - D;
  bool  validCross     = crossedUp && (crossStrength >= 2.0);
  prevK = K; prevD = D;

  // 4) EMA 順位 (1本前の確定バー)
  double eF[1], eM[1], eS[1];
  if(CopyBuffer(hEmaFast,0,1,1,eF)!=1 ||
     CopyBuffer(hEmaMid ,0,1,1,eM)!=1 ||
     CopyBuffer(hEmaSlow,0,1,1,eS)!=1)
    return;
  bool emaLong = (eF[0] > eM[0] && eM[0] > eS[0]);

  // 5) 発注 & ログ出力
  if(armed && emaLong && validCross && !PositionSelect(_Symbol))
  {
    trade.SetExpertMagicNumber(250425);
    if(trade.Buy(0.1, _Symbol))
    {
      ulong ticket = PositionGetInteger(POSITION_TICKET);
      double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double emaGap= eF[0] - eS[0];
      double priceVsFast = ask - eF[0];

      FileWrite(fh,
        TimeToString(TimeCurrent(),TIME_DATE|TIME_SECONDS),
        ticket,
        K, D, crossStrength,
        eF[0], eM[0], eS[0], emaGap,
        ask, priceVsFast
      );
      FileFlush(fh);
      armed = false;  // ワンショット
    }
  }
}

//+------------------------------------------------------------------+
//| 終了処理                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  if(fh != INVALID_HANDLE) FileClose(fh);
}
