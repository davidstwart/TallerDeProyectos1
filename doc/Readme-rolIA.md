# Readme - Rol IA Lab (Encargada de IA) - Extendido

main.py
Propósito: Archivo principal que arranca la aplicación FastAPI y configura la API.
Qué hace:

- Crea la instancia de FastAPI con metadata (título, descripción, versión) y especifica docs/redoc.
- Configura CORS para permitir orígenes y métodos durante el desarrollo.
- Define un manejador global de excepciones para devolver respuestas estructuradas ante errores.
- Define modelos de datos para autenticación (UserRole, RegisterRequest, LoginRequest) aunque las rutas de registro/login están comentadas.
- Importa y registra routers: ia_lab_router, auth_router, root_router y health_router.
  Interacciones:
- Es el punto de entrada de la API. Las peticiones llegan a través de las rutas expuestas por los routers y, en particular, IA Lab, se enrutan a ia_lab_router para su procesamiento.

infrastructure/frameworks/fastapi/ia_lab_router.py
Propósito: Router específico para IA Lab en FastAPI.
Qué hace:

- Define el prefijo de rutas /api/v1/ia-lab y las operaciones disponibles: list_models, generate_dataset, upload_dataset, get_dataset_info, train_model, load_model, predict.
- Usa dependency injection (Depends) para obtener el UseCase IA Lab (IALabUseCase) a través get_use_case(), que crea una instancia con un repositorio en memoria (\_repo).
- Valida entradas y formatea respuestas mediante los DTOs definidos en IA Lab.
- Maneja errores y devuelve respuestas en formato de DTOs de salida.
  Interacciones:
- Actúa como puente entre las rutas HTTP y la lógica de negocio (UseCase). Recibe requests, invoca métodos del UseCase y devuelve resultados estructurados.

application/dto/ia_lab_dto.py
Propósito: Definir DTOs para requests y responses de IA Lab.
Qué hace:

- Define modelos de entrada: GenerateDatasetRequest, TrainModelRequest, PredictRequest, LoadModelRequest.
- Define modelos de salida: DatasetInfoResponse, TrainModelResponse, PredictResponse, UploadCSVResponse, ModelsInfoResponse.
- Incluye ModelName como un Literal con opciones permitidas para modelos (Logistic Regression, Decision Tree, etc.).
- Interacciones:
- Sirve como contrato de serialización/deserialización para las peticiones y respuestas; facilita validación de datos y documentación de la API (OpenAPI).

application/use_cases/ia_lab_use_case.py
Propósito: Implementación de los casos de uso de IA Lab.
Qué hace:

- Mantiene AVAILABLE_MODELS con descripciones y hiperparámetros por modelo.
- init(self, session_repo): recibe un repositorio de sesión (ISessionRepository).
- generate_simulated_dataset(n_samples): genera un dataset sintético, crea un session_id y guarda el DataFrame en el repositorio; devuelve session_id y DatasetInfo.
- upload_csv_dataset(content, filename, target_column): lee CSV desde bytes, valida columnas, crea session_id, guarda el DataFrame y devuelve session_id, lista de columnas y total de filas.
- get_dataset_info(session_id, target_column): obtiene el DataFrame de una sesión y devuelve DatasetInfo.
- train_model(session_id, model_name, params, test_size): prepara X e y, realiza división, normaliza, crea el modelo, entrena, evalúa y persiste el modelo entrenado; devuelve TrainingResult con métricas y outputs.
- load_model(model_id): carga un modelo previamente guardado y crea una nueva sesión para predicción; devuelve session_id.
- predict(session_id, features): carga el modelo de la sesión, transforma las características, realiza predicción y devuelve PredictionResult con predicción y probabilidades si están disponibles.
- Helpers: \_build_simulated_df, \_build_dataset_info, \_build_model para generar datos, crear info de dataset y construir modelos.
- Interacciones:
- Es el motor de negocio que recibe las llamadas desde IA Lab Router y realiza todas las operaciones de manipulación de datos, entrenamiento y predicción, y luego expone resultados empaquetados en dataclasses de dominio para ser convertidos a DTOs por el router.

domain/models/ia_lab_model.py
Propósito: Definir las estructuras de dominio para IA Lab.
Qué hace:
Modelos de datos:
DatasetInfo: total_records, total_features, feature_names, target_classes, statistics, correlation, preview.
TrainingResult: model_name, params, accuracy, precision, recall, f1_score, confusion_matrix, classification_report, feature_names, train_samples, test_samples, session_id.
PredictionResult: prediction, probabilities, model_name, input_features.
LoadedModel: model, model_name, feature_names.
ModelParams: model_name, params, test_size (configuración de modelo).
Interacciones:
Sirven como contenedores de datos que se transfieren entre la capa de negocio y las capas superior/inferior.

application/ports/input/ia_lab_input_port.py
Propósito: Definir la interfaz de los casos de uso (port de entrada).
Qué hace:

- Define métodos abstractos:
- generate_simulated_dataset(n_samples) -> (session_id, DatasetInfo)
- upload_csv_dataset(content, filename, target_column) -> (session_id, columnas, total)
- get_dataset_info(session_id, target_column) -> DatasetInfo
- train_model(session_id, model_name, params, test_size) -> TrainingResult
- load_model(model_id) -> session_id
- predict(session_id, features) -> PredictionResult
- get_available_models() -> lista de modelos y sus hiperparámetros
- Interacciones:
- Sirve como contrato de la capa de presentación hacia la lógica de negocio. IA Lab Router llama a estos métodos sin conocer la implementación.

application/ports/output/ia_lab_output_port.py
Propósito: Definir la interfaz de persistencia (port de salida).
Qué hace:

- Define métodos abstractos:
- save_dataframe(session_id, df, target_column)
- get_dataframe(session_id) -> Optional[(pd.DataFrame, target_column)]
- save_trained_model(session_id, model, scaler, feature_names, model_name)
- persist_model(model_id, model, scaler, feature_names, model_name)
- load_model(model_id) -> Optional[Dict[str, Any]]
- get_trained_model(session_id) -> Optional[Dict[str, Any]]
- session_exists(session_id) -> bool
- list_models() -> list[str]
- Interacciones:
- Proporciona una abstracción para almacenamiento de datasets y modelos entrenados. Permite intercambiar entre memoria, disco o bases de datos sin tocar la lógica de negocio.
