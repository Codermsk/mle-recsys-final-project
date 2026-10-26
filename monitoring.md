# Мониторинг сервиса рекомендаций

Сервис отдаёт метрики в формате Prometheus на `/metrics`. Prometheus опрашивает его каждые
15 секунд (`monitoring/prometheus.yml`), Grafana подключается к Prometheus автоматически и
сразу загружает дашборд «Bank recsys» (`monitoring/grafana/dashboards/recsys.json`).

```bash
docker compose up --build -d
python scripts/simulate_traffic.py --duration 300 --rps 5
```

## Метрики

### Инфраструктурные

Собираются `prometheus-fastapi-instrumentator` и стандартным коллектором процесса.

| Метрика | Тип | Зачем |
|---|---|---|
| `http_requests_total{handler, method, status}` | counter | нагрузка на ручки и доля ошибок |
| `http_request_duration_seconds` | histogram | время ответа (p50, p95) |
| `process_resident_memory_bytes` | gauge | память: хранилище признаков держится в памяти процесса |

### Прикладные

| Метрика | Тип | Зачем |
|---|---|---|
| `recsys_recommendation_requests_total{source}` | counter | сколько клиентов получили рекомендации модели (`model`), а сколько - список популярного (`popular`, клиент не найден в хранилище признаков). Рост доли `popular` значит, что хранилище устарело или пришло много новых клиентов |
| `recsys_recommended_products_total{product}` | counter | как часто каждый продукт попадает в выдачу. Резкое изменение распределения - признак дрейфа данных или проблемы с моделью |
| `recsys_top1_score` | histogram | вероятность первого продукта в списке - уверенность модели. Падение медианы говорит о том, что клиенты стали непохожи на обучающую выборку |
| `recsys_inference_seconds` | histogram | время работы самой модели без HTTP-обвязки |
| `recsys_batch_size` | histogram | сколько клиентов приходит в одном запросе |
| `recsys_client_owned_products` | histogram | число продуктов у клиентов в запросах - контроль входных данных |
| `recsys_model_info{version, features_month}` | gauge | какая модель и какой срез признаков сейчас загружены |
| `recsys_known_clients` | gauge | размер хранилища признаков |

## Дашборд

| Панель | Запрос | На что смотреть |
|---|---|---|
| RPS по ручкам | `sum by (handler) (rate(http_requests_total{handler=~"/recommendations.*"}[1m]))` | нагрузка |
| Латентность p50 / p95 | `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))` | p95 одиночного запроса должен быть меньше 100 мс |
| Доля ошибок 5xx | `sum(rate(http_requests_total{status="5xx"}[5m])) / sum(rate(http_requests_total[5m]))` | должна быть 0 |
| Доля ответов из fallback | доля `source="popular"` | стабильный уровень; рост - повод обновить хранилище признаков |
| Медианный скор top-1 | `histogram_quantile(0.5, sum by (le) (rate(recsys_top1_score_bucket[15m])))` | стабильность уверенности модели |
| Время инференса p95 | `histogram_quantile(0.95, ... recsys_inference_seconds_bucket ...)` | рост при больших пакетах |
| Рекомендованные продукты | `sum by (product) (rate(recsys_recommended_products_total[5m]))` | структура выдачи, дрейф |
| Память процесса | `process_resident_memory_bytes{job="recsys"}` | утечки, рост хранилища |

## Что делать при отклонениях

- **Рост доли fallback** - пересобрать признаки на свежий месяц (`python -m src.train`) и
  перезапустить сервис.
- **Падение скора top-1 или сдвиг распределения продуктов** - сравнить входные данные с обучающими
  (EDA на свежем срезе) и переобучить модель.
- **Рост латентности** - проверить размер пакетов (`recsys_batch_size`) и время инференса; при
  нехватке ресурсов масштабировать сервис горизонтально.
