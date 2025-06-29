#property script_show_inputs
input string CSV = "C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\MQL5\\Files\\signals_fixed.csv";
void OnStart() {
   int h = FileOpen(CSV, FILE_READ|FILE_CSV|FILE_COMMON);
   Print("Handle=",h," Err=",GetLastError());
   int rows = 0;
   if(h!=INVALID_HANDLE){
      FileReadString(h);              // skip header
      while(!FileIsEnding(h)){ FileReadString(h); rows++; }
      FileClose(h);
   }
   Print("Rows=",rows);
}
