import requests
import time
import threading
from prometheus_client import REGISTRY, generate_latest

class GrafanaCloudPusher:
    def __init__(self, remote_write_url, username, api_key):
        self.url = remote_write_url
        self.auth = (username, api_key)
        self.running = False
        self.thread = None

    def push_metrics(self):
        """Envía métricas a Grafana Cloud cada 15 segundos"""
        while self.running:
            try:
                # Obtener métricas actuales
                metrics_data = generate_latest(REGISTRY)

                # Enviar a Grafana Cloud
                response = requests.post(
                    self.url,
                    data=metrics_data,
                    auth=self.auth,
                    headers={'Content-Type': 'application/x-protobuf'},
                    timeout=10
                )

                if response.status_code in [200, 204]:
                    print(f"[{time.strftime('%H:%M:%S')}]  Métricas enviadas a Grafana Cloud")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}]  Error {response.status_code}: {response.text[:100]}")

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}]  Error: {str(e)[:100]}")

            # Esperar 15 segundos
            time.sleep(15)

    def start(self):
        """Inicia el envío automático en background"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.push_metrics, daemon=True)
            self.thread.start()
            print(" Grafana Cloud Pusher iniciado (envío cada 15s)")

    def stop(self):
        """Detiene el envío"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(" Grafana Cloud Pusher detenido")