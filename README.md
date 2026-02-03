

   IA-NDRT


💀 NDRT v3.8: Neural Data Representation Trinity (Mictlan)
      "Más allá del Transformer: Una arquitectura híbrida líquida y lógica creada desde cero."
      
📄 Descripción Ejecutiva

NDRT (Neural Data Representation Trinity) es un sistema de Inteligencia Artificial monolítico de vanguardia que rompe con las limitaciones de las redes neuronales estáticas tradicionales.

A diferencia de los LLMs convencionales, NDRT implementa una arquitectura Neuro-Simbólica Híbrida que fusiona:

  Lógica Estructurada: A través de redes KAN (Kolmogorov-Arnold Networks) de alta fidelidad.
    
  Dinámica Temporal Continua: Utilizando Neural ODEs (Ecuaciones Diferenciales Ordinarias) resueltas mediante el método Runge-Kutta 4 (RK4).
    
  Memoria Trinitaria: Un sistema propietario de gestión de memoria a largo y corto plazo.
  
Este motor, denominado "Mictlan", fue diseñado para operar con precisión mixta (AMP) y escalar en entornos masivos de datos.

🚀 Hazaña de Ingeniería

  🏆 Desarrollado en 1 Mes.Este proyecto es el resultado de un "sprint" de ingeniería intensiva y matemática aplicada. Fue conceptualizado, diseñado y programado en su totalidad en un lapso de solo 30 días, demostrando una capacidad de implementación técnica y visión arquitectónica de alto nivel.
  
  
🧠 Arquitectura Técnica (The Core)NDRT no es un wrapper; es una implementación pura en PyTorch que incluye:

  1. High-Fidelity KAN Layers (Lógica)Implementación personalizada del teorema de representación de Kolmogorov-Arnold. En lugar de pesos fijos en los nodos, NDRT aprende funciones de activación (B-Splines) en las aristas, permitiendo una interpretabilidad lógica y una precisión matemática superior a los MLPs estándar.
  
  2. Liquid Time-Constant Dynamics (Fluidez)El estado oculto de la red evoluciona en el tiempo continuo:
     
     $$\frac{dh}{dt} = -\frac{h}{\tau} + S(x(t))$$

     El sistema utiliza un solver RK4 para estabilizar el aprendizaje en secuencias temporales irregulares, ideal para streams de datos complejos.
  
  3. La "Trinidad" de Memoria
  
  Un sistema jerárquico único diseñado para mitigar el olvido catastrófico:
    
  Memoria HDC (Hyperdimensional Computing): Vectores ortogonales fijos       para protección contra ruido.
     
  Memoria Episódica: Diccionario diferenciable para contexto a corto plazo.
     
  Memoria Cristalizada: Hashing semántico basado en energía para   almacenamiento de conocimiento a largo plazo.

🛠️ Instalación y Uso

Requisitos PreviosPython 3.9+
PyTorch 2.0+ (con soporte CUDA recomendado)
FastAPI & Uvicorn (para modo servidor)

# 1. Clonar el repositorio
git clone https://github.com/fredy1403/IA-NDRT

# 2. Instalar dependencias
pip install torch numpy scipy fastapi uvicorn colorama

# 3. Ejecutar NDRT
python NDRT_V3.8.py

Modos de Ejecución

El sistema cuenta con un menú interactivo CLI que permite:
  Tests de Integridad: Verificar que las matemáticas KAN/ODE funcionan     correctamente.
  Entrenamiento (Streaming): Ingesta de datos en tiempo real.
  Chat Interactivo: Prueba de conversación con el modelo.
  API Server: Despliegue de la API REST para producción.
  
  
⚖️ Licencia y Uso Comercial (Dual Licensing)

  Este proyecto es Software Libre bajo la licencia GNU Affero General Public License v3.0 (AGPLv3).
  
Para la Comunidad (Open Source)
  
  Eres libre de usar, modificar y distribuir este software, siempre y cuando cualquier modificación o servicio en red que utilice NDRT libere su código fuente completo a la comunidad bajo la misma licencia.
  
Para Uso Empresarial (Proprietario / SaaS)
  
  Si deseas implementar NDRT en un entorno comercial cerrado, privado o como parte de un servicio SaaS propietario sin liberar tu código fuente, debes adquirir una Licencia Comercial.
  
El modelo de licenciamiento dual permite a las empresas integrar la potencia de NDRT en sus productos protegiendo su propia propiedad intelectual.

📩 Contacto para Licencias Comerciales:Para consultas sobre precios, integración empresarial y exenciones de la licencia AGPL, contactar directamente desde este medio.

👨‍💻 Autor y Créditos

Este sistema fue creado, diseñado e implementado en su totalidad por: 

Fidel Alfredo Bautista Hernández

  Arquitecto de Software & Investigador de IA
  Matemáticas Aplicadas (KAN & ODEs)
  Implementación de High-Performance Computing
  
  "   GRACIAS POR VER ESTE PROYECTO  "

  
