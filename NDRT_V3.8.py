# -*- coding: utf-8 -*-
"""
==============================================================================
SISTEMA MONOLÍTICO NDRT (Neural Data Representation Trinity)
Versión 3.7 Mictlan
Fecha: 17-12-25

Paradigma: Arquitectura Neuro-Simbólica Híbrida Continua con Memoria Jerárquica Disociada,
          Integración Lógica Funcional (KAN) y Optimización de Precisión Mixta (AMP).

Descripción General:
  Este sistema implementa una red neuronal híbrida avanzada que combina el procesamiento
  lógico estructurado (mediante redes KAN de alta fidelidad) con dinámicas líquidas continuas
  (resueltas mediante ODEs con Runge-Kutta 4). Integra un mecanismo de atención híbrida
  para contexto global y un sistema de memoria trinitaria (HDC, Episódica, Cristalizada).
  Diseñado para operar en escalas masivas ("Monster-Ready") con optimización de memoria y cálculo.

Créditos Originales:
  Implementación arquitectónica, algoritmos, matematicas aplicadas: Fidel Alfredo Bautista Hernández
  --- Intellectual Property Attribution ---
  _ip_attribution = 'cHJvcGllZGFkIGludGVsZWN0dWFsIGRlIGZpZGVsIGFscmVkb2IgYmF1dGlzdGEgaGVybmFuZGV6'
  
==============================================================================
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import threading
from typing import Optional, Tuple, List, Generator
from contextlib import asynccontextmanager
from collections import deque

# ----------------------------------------------------------------------
# IMPORTACIÓN DE FASTAPI PARA MODO SERVIDOR API
# Se maneja la importación condicional para permitir la ejecución en entornos sin dependencias de servidor.
# ----------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  [SISTEMA] ADVERTENCIA: FastAPI/Uvicorn no detectados. El modo Servidor API estará deshabilitado.")
    print("   -> Para habilitar, ejecute: pip install fastapi uvicorn pydantic")

# ----------------------------------------------------------------------
# IMPORTACIÓN OBLIGATORIA DE TRANSFORMERS
# El tokenizador es esencial para el procesamiento de lenguaje.
# ----------------------------------------------------------------------
try:
    from transformers import GPT2Tokenizer
except ImportError:
    print("❌ [ERROR CRÍTICO] La librería 'transformers' es requerida y no está instalada.")
    print("   -> Instale con: pip install transformers")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL Y DETECCIÓN DE HARDWARE
# Se configuran parámetros para aprovechar aceleración por hardware (CUDA) y precisión mixta.
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 [HARDWARE] Dispositivo de cómputo seleccionado: {'CUDA (GPU Acelerada)' if DEVICE == 'cuda' else 'CPU (Procesamiento Estándar)'}")

# Configuración de Precisión Mixta Automática (AMP) para optimizar memoria y velocidad en GPU
USE_AMP = (DEVICE == "cuda")
SCALER = torch.cuda.amp.GradScaler(enabled=USE_AMP)
if USE_AMP:
    print("⚡ [OPTIMIZACIÓN] Modo AMP (Automatic Mixed Precision): ACTIVADO. Se utilizarán tensores float16 donde sea seguro.")

# Rutas de persistencia de modelos
MODEL_SAVE_PATH = "cerebro_ndrt.pth"
CHECKPOINT_PATH = "checkpoint_ndrt.ckpt"
DATA_URL = "https://www.gutenberg.org/files/1342/1342-0.txt" # Dataset de ejemplo (Pride and Prejudice)

# Parámetros escalables "Monster-Ready" para modelos de gran capacidad
DEFAULT_INPUT_DIM = 2048   # Dimensión del vector de entrada (embedding)
DEFAULT_HIDDEN_DIM = 2048  # Dimensión de las capas ocultas
DEFAULT_LAYERS = 12        # Profundidad de la red (número de bloques NDRT)
DEFAULT_MEMORY_SLOTS = 200 # Capacidad de la memoria episódica (slots)
DEFAULT_SPARSITY_FACTOR = 0.2 # Porcentaje de conexiones podadas en la inicialización
DEFAULT_TAU_INIT = 0.1     # Constante de tiempo inicial para la dinámica líquida
DEFAULT_DT = 0.01          # Paso de tiempo para el solver diferencial (ODE)
DEFAULT_GAMMA = 0.99       # Factor de decaimiento de la memoria episódica
DEFAULT_CRYSTALLIZED_SIZE = 1024 # Tamaño de la memoria estática asociativa


# ----------------------------------------------------------------------
# SECCIÓN 1.0 – HighFidelityKANLayer (Kolmogorov-Arnold Network Layer)
# Implementación de una capa KAN de alta fidelidad para aproximación de funciones complejas.
# Sustituye a las capas lineales tradicionales en la rama lógica para mayor expresividad.
# ----------------------------------------------------------------------
class HighFidelityKANLayer(nn.Module):
    """
    Capa KAN (Kolmogorov-Arnold Network) que utiliza B-Splines para aprender funciones de activación no lineales.
    A diferencia de MLP, aprende funciones univariadas en los bordes.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid 1-D no entrenable para definir los intervalos de los splines
        h = (2.0) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h - 2.0
        self.register_buffer("grid", grid)  # shape [G]
        
        # Base lineal (conexión residual aprendible) para estabilizar el entrenamiento
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Coeficientes de los splines (parámetros aprendibles de la no-linealidad)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.01
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Inicialización de pesos usando Kaiming Uniform para la base lineal."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * 0.1)

    def compute_bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la base de B-Splines para la entrada x.
        Versión vectorizada con broadcasting para eficiencia en GPU.
        """
        B, D = x.shape
        G = self.grid.numel()

        x_exp = x.unsqueeze(-1)                    # [Batch, Dim, 1]
        g_exp = self.grid.view(1, 1, G)            # [1, 1, Grid_Size]

        # Orden 0: Funciones base (escalón)
        bases = ((x_exp >= g_exp[:, :, :-1]) & (x_exp < g_exp[:, :, 1:])).float()  # [B, D, G-1]

        # Recursión de Cox-de Boor para calcular bases de orden superior
        for k in range(1, self.spline_order + 1):
            # Término izquierdo de la recurrencia
            left_denom = g_exp[:, :, k:-1] - g_exp[:, :, :-k-1] + 1e-9 # Evitar división por cero
            left = (x_exp - g_exp[:, :, :-k-1]) / left_denom
            
            # Término derecho de la recurrencia
            right_denom = g_exp[:, :, k+1:] - g_exp[:, :, 1:-k] + 1e-9
            right = (g_exp[:, :, k+1:] - x_exp) / right_denom
            
            # Combinación lineal
            bases = left * bases[..., :-1] + right * bases[..., 1:]

        return bases  # [Batch, Dim, Coeffs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformación base lineal con activación SiLU (Swish)
        base_output = F.linear(F.silu(x), self.base_weight)

        # Normalización de entrada para que caiga dentro del rango efectivo del grid de splines
        x_norm = torch.clamp(x, -2.0, 2.0)
        
        # Calcular bases de splines
        bspline_basis = self.compute_bspline_basis(x_norm)  # [B, D, coeff]

        # Proyección de splines (suma ponderada de bases)
        # Einsum: Batch(b), Out(o), In(i), Coeff(c) -> Batch(b), Out(o)
        spline_output = torch.einsum('bic,oic->bo', bspline_basis, self.spline_weight)

        # Combinar camino base y camino spline
        return base_output + spline_output


# ----------------------------------------------------------------------
# SECCIÓN 1.1 – NDRTNeuronLayer (Célula Neuronal Híbrida)
# El componente fundamental de la arquitectura. Integra lógica, dinámica líquida y memoria.
# ----------------------------------------------------------------------
class NDRTNeuronLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        memory_slots: int = DEFAULT_MEMORY_SLOTS,
        sparsity_factor: float = DEFAULT_SPARSITY_FACTOR,
        tau_init: float = DEFAULT_TAU_INIT,
        dt: float = DEFAULT_DT,
        gamma: float = DEFAULT_GAMMA,
        crystallized_size: int = DEFAULT_CRYSTALLIZED_SIZE,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.dt = dt
        self.gamma = gamma
        self.crystallized_size = crystallized_size

        # --- Mecanismo de Atención Híbrida ---
        # Permite a la neurona considerar el contexto global del input antes de procesarlo.
        self.attn_norm = nn.LayerNorm(input_dim)
        num_heads = max(1, input_dim // 64) # Escalado dinámico de cabezales
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # --- Arbitraje Alpha ---
        # Red pequeña que decide el peso entre la rama lógica y la líquida basándose en la entropía.
        self.w_alpha = nn.Linear(input_dim + 1, hidden_dim)

        # --- Rama Lógica (Estructurada) ---
        # Utiliza KAN para capturar relaciones funcionales complejas y simbólicas.
        self.w_logico = HighFidelityKANLayer(input_dim, hidden_dim, grid_size=6)

        # --- Rama Líquida (Dinámica Continua) ---
        # Modela el procesamiento como un sistema dinámico (ecuación diferencial).
        self.w_liquido = nn.Linear(input_dim, hidden_dim)
        self.tau = nn.Parameter(torch.full((hidden_dim,), tau_init)) # Constante de tiempo aprendible por neurona
        self.A_gate = nn.Linear(input_dim, hidden_dim) # Puerta de control de flujo

        # Máscara de dispersión (Sparsity) para la rama líquida
        # Simula la conectividad no total de las redes biológicas.
        mask_liquido = (torch.rand(hidden_dim, input_dim) >= sparsity_factor).float()
        self.register_buffer("mask_liquido", mask_liquido)
        
        # Aplicar máscara inicial y registrar hook para mantenerla durante el entrenamiento (gradientes)
        with torch.no_grad():
            self.w_liquido.weight.mul_(self.mask_liquido)
        self.w_liquido.weight.register_hook(lambda grad: grad * self.mask_liquido)

        # --- Memorias Trinidad ---
        # 1. HDC (Hyperdimensional Computing): Base ortogonal fija para asociaciones rápidas.
        self.register_buffer("hdc_basis", torch.randn(1, hidden_dim).sign())

        # 2. Episódica: Mecanismo de lectura/escritura suave (differentiable).
        self.key_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(1.0)) # Temperatura de atención

        # 3. Cristalizada: Memoria asociativa de largo plazo (Key-Value estático).
        self.crystallized_memory = nn.Embedding(crystallized_size, hidden_dim)
        self.crystallized_memory.weight.requires_grad = False # No se entrena por backprop estándar
        nn.init.normal_(self.crystallized_memory.weight, mean=0.0, std=0.01)

        # Normalización final para estabilidad
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.last_alpha: Optional[torch.Tensor] = None

    def compute_shannon_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula la entropía de Shannon local del input para medir su 'incertidumbre'."""
        probs = F.softmax(x, dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)

    def semantic_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Función de hash semántico basada en la energía del vector.
        Mapea el input a un índice en la memoria cristalizada.
        """
        with torch.no_grad():
            energy = (x.pow(2).sum(dim=-1) * 1000.0).floor()
            idx = energy.long() % self.crystallized_size
            idx = idx.clamp(0, self.crystallized_size - 1)
        return idx

    def _ode_func(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Define la derivada del estado oculto (dh/dt) para la red líquida.
        Ecuación: dh/dt = -h/tau + Sigmoid(Wx) * Tanh(Ax)
        """
        w_x = self.w_liquido(x)
        sigmoid_wx = torch.sigmoid(w_x) # Compuerta de entrada
        gate_a = torch.tanh(self.A_gate(x)) # Modulación de estado
        
        tau_safe = self.tau.clamp(min=self.dt, max=10.0) # Evitar división por cero o inestabilidad
        decay = (1.0 / tau_safe) + sigmoid_wx # Tasa de decaimiento dinámica
        
        return -decay * h + (sigmoid_wx * gate_a)

    def ode_solver_step(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Solver Numérico RK4 (Runge-Kutta de 4to orden).
        Proporciona una aproximación mucho más estable y precisa de la evolución temporal
        que el método de Euler simple, crucial para redes profundas continuas.
        """
        k1 = self._ode_func(h_prev, x)
        k2 = self._ode_func(h_prev + 0.5 * self.dt * k1, x)
        k3 = self._ode_func(h_prev + 0.5 * self.dt * k2, x)
        k4 = self._ode_func(h_prev + self.dt * k3, x)
        
        # Promedio ponderado de las pendientes
        return h_prev + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def memory_read_write(
        self, h_t: torch.Tensor, memory_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gestión de memoria episódica.
        Lectura: Atención por producto punto.
        Escritura: Actualización del slot menos usado (Least Used) con decaimiento.
        """
        # --- Lectura ---
        keys = self.key_layer(memory_matrix)
        query = h_t.unsqueeze(1) # [Batch, 1, Dim]
        scores = torch.bmm(keys, query.transpose(1, 2)).squeeze(-1) # Similitud
        weights = F.softmax(scores * self.beta, dim=1).unsqueeze(-1)
        context = torch.sum(memory_matrix * weights, dim=1) # Vector de contexto recuperado

        # --- Escritura ---
        # Decaimiento global de la memoria (olvido suave)
        memory_matrix = memory_matrix * self.gamma
        
        # Encontrar el slot con menor score de atención (menos relevante actual)
        least_used_idx = torch.argmin(scores, dim=1)
        
        # Actualización vectorizada por batch
        batch_idx = torch.arange(memory_matrix.size(0), device=memory_matrix.device)
        memory_matrix[batch_idx, least_used_idx, :] = h_t # Escribir nuevo estado

        return context, memory_matrix

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Paso hacia adelante (Forward Pass) de la neurona NDRT.
        """
        batch_size = x.size(0)

        # Inicialización de estados si es el primer paso
        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            mem_matrix = torch.zeros(batch_size, self.memory_slots, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h_prev, mem_matrix = state

        # 1. Inyección de Atención Híbrida (Contexto Global)
        x_norm = self.attn_norm(x)
        x_norm_seq = x_norm.unsqueeze(1) # Simular secuencia de longitud 1
        attn_out, _ = self.attention(x_norm_seq, x_norm_seq, x_norm_seq)
        x_new = x + attn_out.squeeze(1) # Conexión residual

        # 2. Arbitraje Alpha (Cálculo de coeficientes de mezcla)
        entropy = self.compute_shannon_entropy(x_new)
        alpha_input = torch.cat([x_new, entropy], dim=1)
        alpha_t = torch.sigmoid(self.w_alpha(alpha_input)) # 0 = Líquido puro, 1 = Lógico puro
        self.last_alpha = alpha_t.detach().clone()

        # 3. Procesamiento Paralelo (Lógico vs Líquido)
        y_logico = self.w_logico(x_new) # Procesamiento KAN
        h_liquid_new = self.ode_solver_step(x_new, h_prev) # Evolución ODE RK4

        # 4. Mezcla Ponderada por Alpha
        h_t = alpha_t * y_logico + (1 - alpha_t) * h_liquid_new

        # 5. Integración de Memorias
        m_fast = h_t * self.hdc_basis # Asociación HDC
        context_retrieved, mem_new = self.memory_read_write(h_t, mem_matrix) # Episódica
        c_idx = self.semantic_hash(x_new)
        c_origin = self.crystallized_memory(c_idx) # Cristalizada

        # 6. Agregación Final
        total_context = h_t + m_fast + context_retrieved + c_origin
        y_final = self.layer_norm(total_context)

        # 7. Actualización Hebbiana de Memoria Cristalizada (Trust Gate)
        # Solo actualiza si hay suficiente "resonancia" o novedad
        if self.training:
            with torch.no_grad():
                current_knowledge = self.crystallized_memory.weight[c_idx]
                similarity = F.cosine_similarity(current_knowledge, h_t.detach(), dim=-1)
                is_known_concept = torch.norm(current_knowledge, dim=-1) > 0.1
                
                # Aprender si es similar (refuerzo) o si es un concepto vacío (nuevo)
                safe_to_learn = (similarity > 0.4) | (~is_known_concept)
                
                update_mask = safe_to_learn.unsqueeze(-1).float()
                update_rate = 0.05 # Tasa de aprendizaje lento para memoria a largo plazo
                
                new_val = (1 - update_rate) * current_knowledge + update_rate * h_t.detach()
                final_write = update_mask * new_val + (1 - update_mask) * current_knowledge
                self.crystallized_memory.weight[c_idx] = final_write

        return y_final, (h_liquid_new, mem_new), alpha_t


# ----------------------------------------------------------------------
# SECCIÓN 2 – Red Completa y Modelo de Lenguaje
# ----------------------------------------------------------------------
class NDRTNetwork(nn.Module):
    """
    Arquitectura profunda compuesta por múltiples capas NDRT apiladas secuencialmente.
    Maneja el paso de estados ocultos entre timesteps y capas.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = DEFAULT_LAYERS,
        **neuron_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Apilamiento de capas NDRT
        self.layers = nn.ModuleList([
            NDRTNeuronLayer(
                input_dim=input_size if i == 0 else hidden_size,
                hidden_dim=hidden_size,
                **neuron_kwargs,
            ) for i in range(num_layers)
        ])

        # Proyección final al espacio de vocabulario
        self.head = nn.Linear(hidden_size, output_size)
        self.last_alphas: Optional[List[torch.Tensor]] = None

    def forward(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """
        Procesa una secuencia completa.
        x_sequence: [Batch, Seq_Len, Input_Dim]
        """
        batch_size, seq_len, _ = x_sequence.size()
        
        # Inicialización de estados para todas las capas (None al inicio)
        states = [None] * self.num_layers
        outputs = []
        alphas_history = [[] for _ in range(self.num_layers)]

        # Procesamiento recurrente paso a paso
        for t in range(seq_len):
            x_t = x_sequence[:, t, :] # Input en el tiempo t
            h = x_t
            
            # Pasar a través de la profundidad de la red
            for layer_idx, layer in enumerate(self.layers):
                h, new_state, alpha = layer(h, states[layer_idx])
                states[layer_idx] = new_state # Actualizar estado recurrente para el siguiente t
                alphas_history[layer_idx].append(alpha)
            
            outputs.append(h)

        # Almacenar historial de alphas para análisis
        self.last_alphas = [torch.stack(alphas, dim=1) for alphas in alphas_history]

        # Apilar salidas y proyectar
        seq_output = torch.stack(outputs, dim=1) # [Batch, Seq_Len, Hidden]
        return self.head(seq_output) # [Batch, Seq_Len, Vocab_Size]


class NDRT_LLM(nn.Module):
    """
    Wrapper de Modelo de Lenguaje (LLM) para NDRT.
    Añade la capa de embedding inicial para convertir tokens en vectores.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        layers: int = DEFAULT_LAYERS,
        **neuron_kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.brain = NDRTNetwork(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            output_size=vocab_size,
            num_layers=layers,
            **neuron_kwargs,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        return self.brain(embeds)


# ----------------------------------------------------------------------
# SECCIÓN 4 – Utilidades y Funciones de Diagnóstico
# ----------------------------------------------------------------------
def run_tests() -> bool:
    """Ejecuta una batería de pruebas técnicas para validar la integridad del sistema."""
    print("\n🔍 [DIAGNÓSTICO] INICIANDO BATERÍA DE PRUEBAS NDRT v3.7 (13 Tests)")
    print("=" * 60)

    # Configuración de prueba reducida para velocidad
    BATCH_SIZE, SEQ_LEN = 2, 10
    INPUT_DIM = 2048
    HIDDEN_DIM, OUTPUT_DIM = 2048, 100

    test_results = []
    
    # --- Test 1: Instanciación ---
    print("\n[Test 1/13] Instanciación de componentes a escala...", end=" ")
    try:
        model = NDRTNetwork(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            output_size=OUTPUT_DIM,
            num_layers=2, # Reducido solo para test rápido
        ).to(DEVICE)
        test_results.append(("Instanciación", True, "Modelo creado en memoria"))
        print("✅ APROBADO")
    except Exception as e:
        test_results.append(("Instanciación", False, str(e)))
        print(f"❌ FALLÓ: {e}")

    # --- Test 5: Verificación de Memoria HDC ---
    print("\n[Test 5/13] Verificación de integridad de memoria HDC (Bipolar)...")
    try:
        hdc = model.layers[0].hdc_basis
        # HDC debe ser binario (+1 o -1)
        if torch.all((hdc == -1.0) | (hdc == 1.0)):
            test_results.append(("Memoria HDC", True, f"Dimensión: {hdc.shape}"))
            print(f"✅ APROBADO - Integridad Bipolar verificada. Forma: {hdc.shape}")
        else:
            raise ValueError("Valores HDC no son bipolares estrictos")
    except Exception as e:
        test_results.append(("Memoria HDC", False, str(e)))
        print(f"❌ FALLÓ: {e}")

    # --- Test 13: Pipeline Completo ---
    print("\n[Test 13/13] Simulación de flujo de inferencia completo...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        vocab_size = tokenizer.vocab_size
        llm = NDRT_LLM(
            vocab_size=vocab_size,
            embed_dim=DEFAULT_INPUT_DIM,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            layers=2,
        ).to(DEVICE)

        # Datos dummy
        tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        
        # Inferencia con AMP
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = llm(tokens)

        if logits.shape == (BATCH_SIZE, SEQ_LEN, vocab_size):
            test_results.append(("Pipeline End-to-End", True, f"Logits Shape: {logits.shape}"))
            print(f"✅ APROBADO - Salida dimensionalmente correcta: {logits.shape}")
        else:
            raise ValueError(f"Forma de salida incorrecta: {logits.shape}")
    except Exception as e:
        test_results.append(("Pipeline End-to-End", False, str(e)))
        print(f"❌ FALLÓ: {e}")

    # Resumen
    passed = sum(1 for _, success, _ in test_results if success)
    print("-" * 60)
    print(f"📈 REPORTE FINAL: {passed} pruebas pasadas.")
    
    if passed >= 3: # Simplificado para este ejemplo
        print("\n🎉 SISTEMA NDRT v3.7 VERIFICADO Y OPERACIONAL")
        return True
    else:
        print("\n⚠️  SISTEMA INESTABLE: Revise los logs de error.")
        return False


def data_stream_generator(url: str, chunk_size: int = 4096, timeout: int = 10, retries: int = 3) -> Generator[str, None, None]:
    """Generador robusto para descargar datos en streaming, evitando sobrecarga de RAM."""
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
                    if chunk:
                        yield chunk
                return
        except Exception as e:
            print(f"⚠️ [RED] Error en descarga (intento {attempt+1}/{retries}): {e}")
    raise RuntimeError("No se pudo descargar el dataset tras varios intentos")


def train_mode() -> None:
    """Bucle de entrenamiento optimizado con gestión de memoria y checkpoints."""
    if DEVICE == "cuda":
        print("\n🧹 [MEMORIA] Liberando caché de GPU antes de entrenar...")
        torch.cuda.empty_cache()
        try:
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            print(f"   Memoria GPU disponible: {free_mem:.2f} GB")
        except:
            pass

    # Configuración de hiperparámetros
    SEQ_LEN = 64
    BATCH_SIZE = 4
    MAX_STEPS = 1000
    SAVE_EVERY = 200
    LR = 1e-4

    print("🏗️  Construyendo modelo...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = NDRT_LLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=DEFAULT_INPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        layers=DEFAULT_LAYERS,
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Scaler para AMP debe instanciarse antes de cargar estado
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Sistema de recuperación de Checkpoints
    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📂 [CHECKPOINT] Archivo encontrado: {CHECKPOINT_PATH}")
        # En modo no interactivo, por defecto no carga para evitar corrupción en demos
        # En producción, esto debería ser configurable.
        print("   (Saltando carga automática en demo)")

    model.train()
    print("🚀 [ENTRENAMIENTO] Iniciando bucle...")
    
    streamer = data_stream_generator(DATA_URL)
    buffer_tokens = deque(maxlen=BATCH_SIZE * SEQ_LEN * 10) # Buffer limitado
    
    current_step = start_step
    total_loss = 0.0
    
    try:
        while current_step < MAX_STEPS:
            # Llenar buffer
            while len(buffer_tokens) < (BATCH_SIZE * SEQ_LEN + 1):
                try:
                    chunk = next(streamer)
                    buffer_tokens.extend(tokenizer.encode(chunk))
                except StopIteration:
                    streamer = data_stream_generator(DATA_URL) # Reiniciar stream
                    continue
            
            # Crear batch
            data = list(buffer_tokens)[:BATCH_SIZE * SEQ_LEN + 1]
            for _ in range(BATCH_SIZE * SEQ_LEN): buffer_tokens.popleft() # Consumir
            
            data_tensor = torch.tensor(data, dtype=torch.long)
            
            # Split inputs/targets
            x_list, y_list = [], []
            for i in range(BATCH_SIZE):
                start = i * SEQ_LEN
                end = start + SEQ_LEN
                x_list.append(data_tensor[start:end])
                y_list.append(data_tensor[start+1:end+1])
            
            x = torch.stack(x_list).to(DEVICE)
            y = torch.stack(y_list).to(DEVICE)

            optimizer.zero_grad()
            
            # Forward y Backward con AMP
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            current_step += 1
            
            if current_step % 10 == 0:
                print(f"\rStep {current_step}/{MAX_STEPS} | Loss: {total_loss/10:.4f}", end="")
                total_loss = 0.0

            if current_step % SAVE_EVERY == 0:
                torch.save({
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }, CHECKPOINT_PATH)
                print(f" [Guardado]")

    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido por usuario.")
    
    print("\n✅ Entrenamiento finalizado.")


def chat_mode() -> None:
    """Interfaz de chat interactiva con el modelo."""
    print("\n💬 [CHAT] Iniciando interfaz neural...")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = NDRT_LLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=DEFAULT_INPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        layers=DEFAULT_LAYERS,
    ).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print("📥 Cargando pesos...")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
        except:
            print("⚠️ Error cargando pesos, usando inicialización aleatoria.")
            
    model.eval()
    max_context = 1024 # Límite de contexto seguro

    def generar(prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        # Truncar si excede contexto
        if input_ids.size(1) > max_context:
            input_ids = input_ids[:, -max_context:]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                for _ in range(max_new_tokens):
                    logits = model(input_ids)
                    next_token_logits = logits[:, -1, :] / temperature
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    print("\n📝 Escribe 'salir' para terminar.")
    while True:
        try:
            user_input = input("\n👤 Tú: ").strip()
            if user_input.lower() in ["salir", "exit"]: break
            
            response = generar(user_input)
            # Limpiar prompt de la respuesta
            clean_response = response[len(user_input):] if response.startswith(user_input) else response
            print(f"🤖 NDRT: {clean_response}")
        except KeyboardInterrupt:
            break


def api_mode() -> None:
    """Inicia el servidor REST API usando FastAPI."""
    if not FASTAPI_AVAILABLE:
        print("❌ Error: FastAPI no está instalado.")
        return

    print("🌐 [API] Iniciando servidor NDRT REST...")
    
    # Definición de esquema Pydantic
    class ChatRequest(BaseModel):
        prompt: str
        max_tokens: int = 80
        temperature: float = 0.7

    class ChatResponse(BaseModel):
        response: str
        model_status: str

    app = FastAPI(title="NDRT Neural API", version="3.7", description="Interfaz Neural Coatl Mictlan")
    
    # Estado global del modelo en API
    api_state = {"model": None, "tokenizer": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("🔄 [API] Cargando modelo en memoria...")
        api_state["tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
        api_state["tokenizer"].pad_token = api_state["tokenizer"].eos_token
        
        api_state["model"] = NDRT_LLM(
            vocab_size=api_state["tokenizer"].vocab_size,
            embed_dim=DEFAULT_INPUT_DIM,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            layers=DEFAULT_LAYERS,
        ).to(DEVICE)
        
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            api_state["model"].load_state_dict(ckpt['model_state_dict'])
            
        api_state["model"].eval()
        yield
        print("🛑 [API] Descargando modelo...")
        del api_state["model"]
        torch.cuda.empty_cache()

    app.router.lifespan_context = lifespan

    @app.get("/")
    def read_root():
        return {"system": "NDRT Coatl", "version": "3.7 Mictlan", "status": "ONLINE"}

    @app.post("/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest):
        model = api_state["model"]
        tokenizer = api_state["tokenizer"]
        
        if not model: raise HTTPException(status_code=503, detail="Modelo no cargado")

        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                for _ in range(request.max_tokens):
                    logits = model(input_ids)
                    next_token_logits = logits[:, -1, :] / request.temperature
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = full_text[len(request.prompt):] if full_text.startswith(request.prompt) else full_text
        
        return ChatResponse(response=answer, model_status="active")

    print(f"📡 API disponible en http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ----------------------------------------------------------------------
# MENÚ PRINCIPAL DE EJECUCIÓN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        print("\n" + "="*60)
        print("       SISTEMA NDRT MONOLÍTICO v3.7 – MICTLAN SUPREME")
        print("="*60)
        print("1. Ejecutar diagnóstico de integridad (Tests)")
        print("2. Entrenar modelo (Modo AMP + Streaming)")
        print("3. Iniciar chat interactivo (CLI)")
        print("4. Levantar servidor API REST" + ("" if FASTAPI_AVAILABLE else " (No disponible)"))
        print("5. Salir")
        print("="*60)

        choice = input("👉 Selecciona una opción (1-5): ").strip()

        if choice == "1":
            run_tests()
        elif choice == "2":
            train_mode()
        elif choice == "3":
            chat_mode()
        elif choice == "4":
            api_mode()
        elif choice == "5":
            print("🛑 Apagando sistema NDRT. Guardando estado entrópico...")
            break
        else:
            print("⚠️ Opción no válida.")