import re
import os

def analyze_btc_logs(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.startswith('trading_bot.log')]
    
    for log_file in sorted(log_files):
        with open(os.path.join(log_dir, log_file), 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        in_btc_block = False
        block = []
        
        for line in lines:
            if '[TRADE ASSET] Processing BTC' in line:
                if in_btc_block:
                    process_block(block)
                block = [line]
                in_btc_block = True
            elif in_btc_block:
                block.append(line)
                if '----------------------------------------------------------------------' in line and len(block) > 1:
                    process_block(block)
                    in_btc_block = False
                    block = []
        if in_btc_block:
            process_block(block)

def process_block(block):
    decision = "N/A"
    total_score = "N/A"
    buy_score = "N/A"
    sell_score = "N/A"
    reason = "N/A"
    trend_score = "N/A"
    
    in_council_members = False
    for line in block:
        if "COUNCIL MEMBERS" in line:
            in_council_members = True
        
        if in_council_members and "TREND" in line and "pts" in line:
            trend_match = re.search(r'\s*\d\.\s+TREND\s+\(([\d\.]+) pts\)', line)
            if trend_match:
                trend_score = trend_match.group(1)
                in_council_members = False

        if "[COUNCIL] Decision:" in line:
            decision_match = re.search(r'Decision: (.*)', line)
            if decision_match:
                decision = decision_match.group(1).strip()
        
        if "Total Score:" in line:
            score_match = re.search(r'Total Score: ([\d\.]+)/\d+', line)
            if score_match:
                total_score = score_match.group(1)

        if "BUY:" in line and "SELL:" in line:
            buy_sell_match = re.search(r'BUY: ([\d\.]+), SELL: ([\d\.]+)', line)
            if buy_sell_match:
                buy_score = buy_sell_match.group(1)
                sell_score = buy_sell_match.group(2)
        
        if "[RISK] REJECTED" in line:
            reason = line.strip()

    if decision != "N/A":
        print(f"--- BTC Decision Block ---")
        print(f"Decision: {decision}")
        print(f"Total Score: {total_score}")
        print(f"Buy/Sell Scores: {buy_score}/{sell_score}")
        print(f"Trend Score: {trend_score}")
        if reason != "N/A":
            print(f"Reason for no execution: {reason}")
        print("-" * 25)

if __name__ == "__main__":
    analyze_btc_logs("logs/")