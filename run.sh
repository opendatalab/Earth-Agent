#!/bin/bash
wait_for_port() {
  port=$1
  url="http://localhost:${port}/mcp"
  timeout=30
  count=0

  while true; do
    if curl -s "$url" >/dev/null 2>&1; then
      echo "Port $port ($url) is ready!"
      break
    fi

    sleep 1
    count=$((count + 1))
    if [ $count -ge $timeout ]; then
      echo "❌ Timeout waiting for $url"
      exit 1
    fi
    echo "Waiting for $url..."
  done
}
cleanup() {
    echo "Cleanning..."
    for port in 16000 16001 16002 16003 16004 16005 16006 16007 20000 20001 20002 20003 20004; do
        lsof -ti tcp:${port} | xargs -r kill -9
    done
    pkill -9 -f "uvicorn" 2>/dev/null || true
    pkill -9 -f "tools/Analysis.py" 2>/dev/null || true
    pkill -9 -f "tools/Index.py" 2>/dev/null || true
    pkill -9 -f "tools/Inversion.py" 2>/dev/null || true
    pkill -9 -f "tools/Statistics.py" 2>/dev/null || true

    echo "Done"
}


trap cleanup SIGINT SIGTERM EXIT


cd /path/to/EarthAgent-Online

# for insam
export OPENAI_API_KEY='sk-xxx'
export OPENAI_BASE_URL=''

# Start servers in background
bash tools/run_servers.sh 2>/dev/null &
SERVERS_PID=$!
sleep 60

wait_for_port 20000
wait_for_port 20001
wait_for_port 20002
wait_for_port 20003
wait_for_port 20004

# Run the agent
conda run -n fastapi python run_agent.py
