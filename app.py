import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Cálculo Vetorial - PPGMNE/UFPR",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Laboratório Virtual de Operadores Diferenciais Vetoriais")
st.markdown(
    "**UFPR — Programa de Pós-Graduação em Métodos Numéricos em Engenharia (PPGMNE)**  \n"
    "*Prática de Docência 2026/2 — Ferramenta interativa para resolução analítica passo a passo, "
    "avaliação numérica e visualização 2D/3D de Gradiente, Divergente, Rotacional (Curl), Laplaciano e Operadores Mistos.*"
)
st.divider()

# Variáveis simbólicas
x, y, z = sp.symbols('x y z', real=True)

# ==============================================================================
# FUNÇÕES AUXILIARES DE CÁLCULO SIMBÓLICO E NUMÉRICO
# ==============================================================================
def parse_expr(expr_str):
    """Converte string do usuário em expressão SymPy de forma segura."""
    expr_clean = expr_str.replace("^", "**")
    local_dict = {
        'x': x, 'y': y, 'z': z,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
        'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E
    }
    return sp.sympify(expr_clean, locals=local_dict)

def calc_gradient(f):
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    df_dz = sp.diff(f, z)
    return [sp.simplify(df_dx), sp.simplify(df_dy), sp.simplify(df_dz)], (df_dx, df_dy, df_dz)

def calc_divergence(F):
    dFx_dx = sp.diff(F[0], x)
    dFy_dy = sp.diff(F[1], y)
    dFz_dz = sp.diff(F[2], z)
    div_res = sp.simplify(dFx_dx + dFy_dy + dFz_dz)
    return div_res, (dFx_dx, dFy_dy, dFz_dz)

def calc_curl(F):
    dFz_dy = sp.diff(F[2], y)
    dFy_dz = sp.diff(F[1], z)
    dFx_dz = sp.diff(F[0], z)
    dFz_dx = sp.diff(F[2], x)
    dFy_dx = sp.diff(F[1], x)
    dFx_dy = sp.diff(F[0], y)
    cx = sp.simplify(dFz_dy - dFy_dz)
    cy = sp.simplify(dFx_dz - dFz_dx)
    cz = sp.simplify(dFy_dx - dFx_dy)
    return [cx, cy, cz], (dFz_dy, dFy_dz, dFx_dz, dFz_dx, dFy_dx, dFx_dy)

def calc_laplacian_scalar(f):
    d2f_dx2 = sp.diff(f, x, 2)
    d2f_dy2 = sp.diff(f, y, 2)
    d2f_dz2 = sp.diff(f, z, 2)
    lap_res = sp.simplify(d2f_dx2 + d2f_dy2 + d2f_dz2)
    return lap_res, (d2f_dx2, d2f_dy2, d2f_dz2)

def eval_on_grid(expr, X, Y, Z):
    """Avalia expressão SymPy em uma malha NumPy garantindo compatibilidade com constantes."""
    func = sp.lambdify((x, y, z), expr, modules=['numpy'])
    with np.errstate(divide='ignore', invalid='ignore'):
        res = func(X, Y, Z)
        if np.isscalar(res) or (isinstance(res, np.ndarray) and res.ndim == 0):
            res = np.full_like(X, float(res), dtype=float)
        else:
            res = np.array(res, dtype=float)
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

# ==============================================================================
# BARRA LATERAL: SELEÇÃO DE PROBLEMA E ENTRADA DE DADOS
# ==============================================================================
st.sidebar.header("⚙️ Configuração do Problema")

preset = st.sidebar.selectbox(
    "📚 Exemplos Didáticos Pré-definidos:",
    [
        "Personalizado (Entrada Livre)",
        "1. Potencial Gravitacional/Elétrico 2D (Gradiente)",
        "2. Potencial Paraboloide 3D (Gradiente / Laplaciano)",
        "3. Fonte Radial (Divergente > 0)",
        "4. Sorvedouro Radial (Divergente < 0)",
        "5. Vórtice / Redemoinho (Rotacional / Curl)",
        "6. Campo Misto: Vórtice com Fonte",
        "7. Identidade Mista: Rotacional do Gradiente",
        "8. Identidade Mista: Divergente do Rotacional"
    ]
)

# Valores padrão baseados no exemplo selecionado
default_f = "x**2 + y**2 + z**2"
default_Fx, default_Fy, default_Fz = "x", "y", "z"
default_op_idx = 0

if "1. Potencial" in preset:
    default_f = "-1 / sqrt(x**2 + y**2 + 0.5)"
    default_op_idx = 0
elif "2. Potencial" in preset:
    default_f = "x**2 + y**2 - 2*z**2"
    default_op_idx = 3
elif "3. Fonte" in preset:
    default_Fx, default_Fy, default_Fz = "x", "y", "z"
    default_op_idx = 1
elif "4. Sorvedouro" in preset:
    default_Fx, default_Fy, default_Fz = "-x", "-y", "-z"
    default_op_idx = 1
elif "5. Vórtice" in preset:
    default_Fx, default_Fy, default_Fz = "-y", "x", "0"
    default_op_idx = 2
elif "6. Campo Misto" in preset:
    default_Fx, default_Fy, default_Fz = "0.5*x - y", "x + 0.5*y", "0"
    default_op_idx = 2
elif "7. Identidade" in preset:
    default_f = "x**2 * y + sin(y*z) + exp(x)"
    default_op_idx = 6
elif "8. Identidade" in preset:
    default_Fx, default_Fy, default_Fz = "x*y*z", "x**2 + z", "sin(x*y)"
    default_op_idx = 7

operadores_lista = [
    "Gradiente: ∇f (Campo Escalar → Vetorial)",
    "Divergente: ∇·F (Campo Vetorial → Escalar)",
    "Rotacional (Curl): ∇×F (Campo Vetorial → Vetorial)",
    "Laplaciano Escalar: ∇²f (Campo Escalar → Escalar)",
    "Laplaciano Vetorial: ∇²F (Campo Vetorial → Vetorial)",
    "Misto: Divergente do Gradiente ∇·(∇f) = ∇²f",
    "Misto: Rotacional do Gradiente ∇×(∇f)",
    "Misto: Divergente do Rotacional ∇·(∇×F)",
    "Misto: Gradiente do Divergente ∇(∇·F)",
    "Misto: Rotacional do Rotacional ∇×(∇×F)",
    "Misto Escalar-Vetorial: Divergente de Produto ∇·(f·F)"
]

operador = st.sidebar.selectbox(
    "🧮 Selecione o Operador / Cálculo:",
    operadores_lista,
    index=default_op_idx
)

# Determina quais caixas de texto exibir
needs_scalar = any(k in operador for k in ["∇f", "∇²f", "f·F"])
needs_vector = any(k in operador for k in ["F", "∇²F"])

st.sidebar.subheader("📝 Funções Matemáticas Originais")
st.sidebar.caption("Use `x`, `y`, `z`, `**` ou `^` para potência, `sin`, `cos`, `exp`, `sqrt`, `log`.")

if needs_scalar:
    f_str = st.sidebar.text_input("Função Escalar f(x, y, z):", value=default_f)
else:
    f_str = default_f

if needs_vector:
    Fx_str = st.sidebar.text_input("Componente P = Fx(x, y, z) [î]:", value=default_Fx)
    Fy_str = st.sidebar.text_input("Componente Q = Fy(x, y, z) [ĵ]:", value=default_Fy)
    Fz_str = st.sidebar.text_input("Componente R = Fz(x, y, z) [k̂]:", value=default_Fz)
else:
    Fx_str, Fy_str, Fz_str = default_Fx, default_Fy, default_Fz

st.sidebar.subheader("📍 Ponto para Avaliação Numérica")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)
x0 = col_p1.number_input("x₀", value=1.0, step=0.5)
y0 = col_p2.number_input("y₀", value=1.0, step=0.5)
z0 = col_p3.number_input("z₀", value=1.0, step=0.5)

st.sidebar.subheader("📊 Parâmetros de Visualização")
L_dom = st.sidebar.slider("Domínio Espacial [-L, L]:", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
z_slice = st.sidebar.slider("Plano de Corte 2D (z fixo):", min_value=-L_dom, max_value=L_dom, value=float(z0), step=0.5)

# ==============================================================================
# PROCESSAMENTO E RESOLUÇÃO PASSO A PASSO
# ==============================================================================
try:
    f_sym = parse_expr(f_str) if needs_scalar else sp.Integer(0)
    F_sym = [parse_expr(Fx_str), parse_expr(Fy_str), parse_expr(Fz_str)] if needs_vector else [sp.Integer(0)]*3
except Exception as e:
    st.error(f"Erro ao interpretar a expressão matemática: {e}. Verifique os parênteses e a sintaxe.")
    st.stop()

subs_pt = {x: x0, y: y0, z: z0}

# Layout principal em abas
tab_analitica, tab_graf2d, tab_graf3d = st.tabs([
    "📖 1. Solução Analítica Passo a Passo e Resultado Numérico",
    "🗺️ 2. Representação Gráfica 2D (Plano XY)",
    "🧊 3. Representação Gráfica 3D Interativa (Espaço XYZ)"
])

# Variáveis para os gráficos
plot_scalar_expr = None
plot_scalar_title = ""
plot_vec_expr = None
plot_vec_title = ""

with tab_analitica:
    st.subheader("1️⃣ Formulação do Problema")
    if needs_scalar:
        st.latex(rf"f(x, y, z) = {sp.latex(f_sym)}")
    if needs_vector:
        st.latex(
            rf"\vec{{F}}(x, y, z) = \left({sp.latex(F_sym[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(F_sym[1])}\right)\hat{{j}} + "
            rf"\left({sp.latex(F_sym[2])}\right)\hat{{k}}"
        )

    st.subheader("2️⃣ Desenvolvimento Analítico Passo a Passo (Equação Simbólica)")

    # --------------------------------------------------------------------------
    # CASO 1: GRADIENTE
    # --------------------------------------------------------------------------
    if operador.startswith("Gradiente:"):
        grad_res, (dfx, dfy, dfz) = calc_gradient(f_sym)
        st.markdown("**Passo 1 — Definição do Operador Gradiente:**")
        st.latex(r"\nabla f = \frac{\partial f}{\partial x}\hat{i} + \frac{\partial f}{\partial y}\hat{j} + \frac{\partial f}{\partial z}\hat{k}")

        st.markdown("**Passo 2 — Cálculo das Derivadas Parciais de 1ª Ordem:**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = \frac{{\partial}}{{\partial x}}\left({sp.latex(f_sym)}\right) = {sp.latex(dfx)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = \frac{{\partial}}{{\partial y}}\left({sp.latex(f_sym)}\right) = {sp.latex(dfy)}")
        st.latex(rf"\frac{{\partial f}}{{\partial z}} = \frac{{\partial}}{{\partial z}}\left({sp.latex(f_sym)}\right) = {sp.latex(dfz)}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante (antes da substituição numérica):**")
        st.latex(
            rf"\nabla f(x, y, z) = \left({sp.latex(grad_res[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(grad_res[1])}\right)\hat{{j}} + "
            rf"\left({sp.latex(grad_res[2])}\right)\hat{{k}}"
        )
        res_type = "vector"
        res_sym = grad_res
        plot_scalar_expr = f_sym
        plot_scalar_title = "Campo Escalar Original f(x,y) [Potencial]"
        plot_vec_expr = grad_res
        plot_vec_title = "Campo Vetorial Gradiente ∇f (Direção de Máxima Taxa de Crescimento)"

    # --------------------------------------------------------------------------
    # CASO 2: DIVERGENTE
    # --------------------------------------------------------------------------
    elif operador.startswith("Divergente:"):
        div_res, (dFx, dFy, dFz) = calc_divergence(F_sym)
        st.markdown("**Passo 1 — Definição do Operador Divergente:**")
        st.latex(r"\nabla \cdot \vec{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}")

        st.markdown("**Passo 2 — Cálculo das Derivadas Parciais de cada Componente:**")
        st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = \frac{{\partial}}{{\partial x}}\left({sp.latex(F_sym[0])}\right) = {sp.latex(dFx)}")
        st.latex(rf"\frac{{\partial F_y}}{{\partial y}} = \frac{{\partial}}{{\partial y}}\left({sp.latex(F_sym[1])}\right) = {sp.latex(dFy)}")
        st.latex(rf"\frac{{\partial F_z}}{{\partial z}} = \frac{{\partial}}{{\partial z}}\left({sp.latex(F_sym[2])}\right) = {sp.latex(dFz)}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante (antes da substituição numérica):**")
        st.latex(rf"\nabla \cdot \vec{{F}}(x, y, z) = \left({sp.latex(dFx)}\right) + \left({sp.latex(dFy)}\right) + \left({sp.latex(dFz)}\right) = {sp.latex(div_res)}")

        res_type = "scalar"
        res_sym = div_res
        plot_scalar_expr = div_res
        plot_scalar_title = "Divergente ∇·F (Mapa de Fonte > 0 / Sorvedouro < 0)"
        plot_vec_expr = F_sym
        plot_vec_title = "Campo Vetorial Original F(x,y)"

    # --------------------------------------------------------------------------
    # CASO 3: ROTACIONAL (CURL)
    # --------------------------------------------------------------------------
    elif operador.startswith("Rotacional (Curl):"):
        curl_res, (dFz_dy, dFy_dz, dFx_dz, dFz_dx, dFy_dx, dFx_dy) = calc_curl(F_sym)
        st.markdown("**Passo 1 — Definição do Operador Rotacional (Determinante Simbólico):**")
        st.latex(
            r"\nabla \times \vec{F} = \begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ "
            r"\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ "
            r"F_x & F_y & F_z \end{vmatrix} = "
            r"\left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right)\hat{i} + "
            r"\left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}\right)\hat{j} + "
            r"\left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)\hat{k}"
        )

        st.markdown("**Passo 2 — Cálculo das Derivadas Parciais Cruzadas:**")
        st.latex(rf"\text{{Componente }}\hat{{i}}: \quad \frac{{\partial F_z}}{{\partial y}} - \frac{{\partial F_y}}{{\partial z}} = \left({sp.latex(dFz_dy)}\right) - \left({sp.latex(dFy_dz)}\right) = {sp.latex(curl_res[0])}")
        st.latex(rf"\text{{Componente }}\hat{{j}}: \quad \frac{{\partial F_x}}{{\partial z}} - \frac{{\partial F_z}}{{\partial x}} = \left({sp.latex(dFx_dz)}\right) - \left({sp.latex(dFz_dx)}\right) = {sp.latex(curl_res[1])}")
        st.latex(rf"\text{{Componente }}\hat{{k}}: \quad \frac{{\partial F_y}}{{\partial x}} - \frac{{\partial F_x}}{{\partial y}} = \left({sp.latex(dFy_dx)}\right) - \left({sp.latex(dFx_dy)}\right) = {sp.latex(curl_res[2])}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante (antes da substituição numérica):**")
        st.latex(
            rf"\nabla \times \vec{{F}}(x, y, z) = \left({sp.latex(curl_res[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(curl_res[1])}\right)\hat{{j}} + "
            rf"\left({sp.latex(curl_res[2])}\right)\hat{{k}}"
        )
        res_type = "vector"
        res_sym = curl_res
        plot_scalar_expr = curl_res[2]
        plot_scalar_title = "Componente Z do Rotacional (∇×F)_z [Intensidade do Vórtice 2D]"
        plot_vec_expr = F_sym
        plot_vec_title = "Campo Vetorial Original F(x,y) [Vórtice / Redemoinho]"

    # --------------------------------------------------------------------------
    # CASO 4: LAPLACIANO ESCALAR
    # --------------------------------------------------------------------------
    elif operador.startswith("Laplaciano Escalar:") or "Divergente do Gradiente" in operador:
        grad_res, (dfx, dfy, dfz) = calc_gradient(f_sym)
        lap_res, (d2x, d2y, d2z) = calc_laplacian_scalar(f_sym)
        st.markdown("**Passo 1 — Definição do Operador Laplaciano Escalar ($\nabla^2 f = \nabla \cdot (\nabla f)$):**")
        st.latex(r"\nabla^2 f = \nabla \cdot (\nabla f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}")

        st.markdown("**Passo 2 — Derivadas Parciais de 1ª Ordem (Gradiente $\nabla f$):**")
        st.latex(rf"\nabla f = \left({sp.latex(dfx)}\right)\hat{{i}} + \left({sp.latex(dfy)}\right)\hat{{j}} + \left({sp.latex(dfz)}\right)\hat{{k}}")

        st.markdown("**Passo 3 — Derivadas Parciais de 2ª Ordem (Divergente do Gradiente):**")
        st.latex(rf"\frac{{\partial^2 f}}{{\partial x^2}} = {sp.latex(d2x)}, \quad \frac{{\partial^2 f}}{{\partial y^2}} = {sp.latex(d2y)}, \quad \frac{{\partial^2 f}}{{\partial z^2}} = {sp.latex(d2z)}")

        st.markdown("**Passo 4 — Equação Simbólica Resultante (antes da substituição numérica):**")
        st.latex(rf"\nabla^2 f(x, y, z) = {sp.latex(lap_res)}")

        res_type = "scalar"
        res_sym = lap_res
        plot_scalar_expr = lap_res
        plot_scalar_title = "Laplaciano ∇²f(x,y)"
        plot_vec_expr = grad_res
        plot_vec_title = "Gradiente Intermediário ∇f(x,y)"

    # --------------------------------------------------------------------------
    # CASO 5: LAPLACIANO VETORIAL
    # --------------------------------------------------------------------------
    elif operador.startswith("Laplaciano Vetorial:"):
        lap_Fx, _ = calc_laplacian_scalar(F_sym[0])
        lap_Fy, _ = calc_laplacian_scalar(F_sym[1])
        lap_Fz, _ = calc_laplacian_scalar(F_sym[2])
        lap_vec = [lap_Fx, lap_Fy, lap_Fz]

        st.markdown("**Passo 1 — Definição do Operador Laplaciano Vetorial em Coordenadas Cartesianas:**")
        st.latex(r"\nabla^2 \vec{F} = \left(\nabla^2 F_x\right)\hat{i} + \left(\nabla^2 F_y\right)\hat{j} + \left(\nabla^2 F_z\right)\hat{k}")

        st.markdown("**Passo 2 — Laplaciano Escalar de cada Componente:**")
        st.latex(rf"\nabla^2 F_x = {sp.latex(lap_Fx)}")
        st.latex(rf"\nabla^2 F_y = {sp.latex(lap_Fy)}")
        st.latex(rf"\nabla^2 F_z = {sp.latex(lap_Fz)}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante:**")
        st.latex(
            rf"\nabla^2 \vec{{F}}(x, y, z) = \left({sp.latex(lap_Fx)}\right)\hat{{i}} + "
            rf"\left({sp.latex(lap_Fy)}\right)\hat{{j}} + \left({sp.latex(lap_Fz)}\right)\hat{{k}}"
        )
        res_type = "vector"
        res_sym = lap_vec
        plot_scalar_expr = sp.sqrt(lap_Fx**2 + lap_Fy**2 + lap_Fz**2)
        plot_scalar_title = "Magnitude do Laplaciano Vetorial |∇²F|"
        plot_vec_expr = lap_vec
        plot_vec_title = "Campo Laplaciano Vetorial ∇²F"

    # --------------------------------------------------------------------------
    # CASO 6: MISTO - ROTACIONAL DO GRADIENTE
    # --------------------------------------------------------------------------
    elif "Rotacional do Gradiente" in operador:
        grad_res, _ = calc_gradient(f_sym)
        curl_grad, (dFz_dy, dFy_dz, dFx_dz, dFz_dx, dFy_dx, dFx_dy) = calc_curl(grad_res)

        st.markdown("**Passo 1 — Etapa Intermediária: Cálculo do Gradiente $\vec{G} = \nabla f$:**")
        st.latex(
            rf"\vec{{G}} = \nabla f = \left({sp.latex(grad_res[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(grad_res[1])}\right)\hat{{j}} + \left({sp.latex(grad_res[2])}\right)\hat{{k}}"
        )

        st.markdown("**Passo 2 — Aplicação do Rotacional sobre $\vec{G}$ ($\nabla \times (\nabla f)$):**")
        st.latex(rf"\text{{Componente }}\hat{{i}}: \frac{{\partial G_z}}{{\partial y}} - \frac{{\partial G_y}}{{\partial z}} = \left({sp.latex(dFz_dy)}\right) - \left({sp.latex(dFy_dz)}\right) = {sp.latex(curl_grad[0])}")
        st.latex(rf"\text{{Componente }}\hat{{j}}: \frac{{\partial G_x}}{{\partial z}} - \frac{{\partial G_z}}{{\partial x}} = \left({sp.latex(dFx_dz)}\right) - \left({sp.latex(dFz_dx)}\right) = {sp.latex(curl_grad[1])}")
        st.latex(rf"\text{{Componente }}\hat{{k}}: \frac{{\partial G_y}}{{\partial x}} - \frac{{\partial G_x}}{{\partial y}} = \left({sp.latex(dFy_dx)}\right) - \left({sp.latex(dFx_dy)}\right) = {sp.latex(curl_grad[2])}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante (Teorema de Clairaut — Campo Conservativo é Irrotacional):**")
        st.latex(rf"\nabla \times (\nabla f) = \left({sp.latex(curl_grad[0])}\right)\hat{{i}} + \left({sp.latex(curl_grad[1])}\right)\hat{{j}} + \left({sp.latex(curl_grad[2])}\right)\hat{{k}}")

        res_type = "vector"
        res_sym = curl_grad
        plot_scalar_expr = f_sym
        plot_scalar_title = "Potencial Escalar f(x,y)"
        plot_vec_expr = grad_res
        plot_vec_title = "Campo Conservativo ∇f (Irrotacional: ∇×(∇f) = 0)"

    # --------------------------------------------------------------------------
    # CASO 7: MISTO - DIVERGENTE DO ROTACIONAL
    # --------------------------------------------------------------------------
    elif "Divergente do Rotacional" in operador:
        curl_res, _ = calc_curl(F_sym)
        div_curl, (dCx, dCy, dCz) = calc_divergence(curl_res)

        st.markdown("**Passo 1 — Etapa Intermediária: Cálculo do Rotacional $\vec{C} = \nabla \times \vec{F}$:**")
        st.latex(
            rf"\vec{{C}} = \nabla \times \vec{{F}} = \left({sp.latex(curl_res[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(curl_res[1])}\right)\hat{{j}} + \left({sp.latex(curl_res[2])}\right)\hat{{k}}"
        )

        st.markdown("**Passo 2 — Aplicação do Divergente sobre $\vec{C}$ ($\nabla \cdot (\nabla \times \vec{F})$):**")
        st.latex(rf"\frac{{\partial C_x}}{{\partial x}} = {sp.latex(dCx)}, \quad \frac{{\partial C_y}}{{\partial y}} = {sp.latex(dCy)}, \quad \frac{{\partial C_z}}{{\partial z}} = {sp.latex(dCz)}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante (Identidade Vetorial Solenoidal):**")
        st.latex(rf"\nabla \cdot (\nabla \times \vec{{F}}) = \left({sp.latex(dCx)}\right) + \left({sp.latex(dCy)}\right) + \left({sp.latex(dCz)}\right) = {sp.latex(div_curl)}")

        res_type = "scalar"
        res_sym = div_curl
        plot_scalar_expr = curl_res[2]
        plot_scalar_title = "Componente Z do Rotacional Intermediário (∇×F)_z"
        plot_vec_expr = curl_res
        plot_vec_title = "Campo Rotacional Intermediário C = ∇×F (Solenoidal: ∇·C = 0)"

    # --------------------------------------------------------------------------
    # CASO 8: MISTO - GRADIENTE DO DIVERGENTE
    # --------------------------------------------------------------------------
    elif "Gradiente do Divergente" in operador:
        div_res, _ = calc_divergence(F_sym)
        grad_div, (gdx, gdy, gdz) = calc_gradient(div_res)

        st.markdown("**Passo 1 — Etapa Intermediária: Cálculo do Divergente $D(x,y,z) = \nabla \cdot \vec{F}$:**")
        st.latex(rf"D(x, y, z) = \nabla \cdot \vec{{F}} = {sp.latex(div_res)}")

        st.markdown("**Passo 2 — Aplicação do Gradiente sobre $D(x,y,z)$ ($\nabla(\nabla \cdot \vec{F})$):**")
        st.latex(rf"\frac{{\partial D}}{{\partial x}} = {sp.latex(gdx)}, \quad \frac{{\partial D}}{{\partial y}} = {sp.latex(gdy)}, \quad \frac{{\partial D}}{{\partial z}} = {sp.latex(gdz)}")

        st.markdown("**Passo 3 — Equação Simbólica Resultante:**")
        st.latex(
            rf"\nabla(\nabla \cdot \vec{{F}}) = \left({sp.latex(grad_div[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(grad_div[1])}\right)\hat{{j}} + \left({sp.latex(grad_div[2])}\right)\hat{{k}}"
        )
        res_type = "vector"
        res_sym = grad_div
        plot_scalar_expr = div_res
        plot_scalar_title = "Divergente Intermediário D = ∇·F"
        plot_vec_expr = grad_div
        plot_vec_title = "Gradiente do Divergente ∇(∇·F)"

    # --------------------------------------------------------------------------
    # CASO 9: MISTO - ROTACIONAL DO ROTACIONAL
    # --------------------------------------------------------------------------
    elif "Rotacional do Rotacional" in operador:
        curl1, _ = calc_curl(F_sym)
        curl2, _ = calc_curl(curl1)

        st.markdown("**Passo 1 — Etapa Intermediária: Primeiro Rotacional $\vec{C} = \nabla \times \vec{F}$:**")
        st.latex(
            rf"\vec{{C}} = \nabla \times \vec{{F}} = \left({sp.latex(curl1[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(curl1[1])}\right)\hat{{j}} + \left({sp.latex(curl1[2])}\right)\hat{{k}}"
        )

        st.markdown("**Passo 2 — Segundo Rotacional $\nabla \times \vec{C} = \nabla \times (\nabla \times \vec{F}) = \nabla(\nabla \cdot \vec{F}) - \nabla^2 \vec{F}$:**")
        st.latex(
            rf"\nabla \times (\nabla \times \vec{{F}}) = \left({sp.latex(curl2[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(curl2[1])}\right)\hat{{j}} + \left({sp.latex(curl2[2])}\right)\hat{{k}}"
        )
        res_type = "vector"
        res_sym = curl2
        plot_scalar_expr = sp.sqrt(curl2[0]**2 + curl2[1]**2 + curl2[2]**2)
        plot_scalar_title = "Magnitude |∇×(∇×F)|"
        plot_vec_expr = curl2
        plot_vec_title = "Campo Resultante ∇×(∇×F)"

    # --------------------------------------------------------------------------
    # CASO 10: MISTO ESCALAR-VETORIAL - DIVERGENTE DE PRODUTO
    # --------------------------------------------------------------------------
    elif "Divergente de Produto" in operador:
        prod_vec = [sp.simplify(f_sym * F_sym[0]), sp.simplify(f_sym * F_sym[1]), sp.simplify(f_sym * F_sym[2])]
        div_prod, (dPx, dPy, dPz) = calc_divergence(prod_vec)

        st.markdown("**Passo 1 — Identidade do Divergente de Escalar por Vetor:**")
        st.latex(r"\nabla \cdot (f\vec{F}) = (\nabla f) \cdot \vec{F} + f(\nabla \cdot \vec{F})")

        st.markdown("**Passo 2 — Campo Vetorial Composto $\vec{G} = f\vec{F}$:**")
        st.latex(
            rf"\vec{{G}} = \left({sp.latex(prod_vec[0])}\right)\hat{{i}} + "
            rf"\left({sp.latex(prod_vec[1])}\right)\hat{{j}} + \left({sp.latex(prod_vec[2])}\right)\hat{{k}}"
        )

        st.markdown("**Passo 3 — Equação Simbólica Resultante:**")
        st.latex(rf"\nabla \cdot (f\vec{{F}}) = \left({sp.latex(dPx)}\right) + \left({sp.latex(dPy)}\right) + \left({sp.latex(dPz)}\right) = {sp.latex(div_prod)}")

        res_type = "scalar"
        res_sym = div_prod
        plot_scalar_expr = div_prod
        plot_scalar_title = "Divergente Composto ∇·(f·F)"
        plot_vec_expr = prod_vec
        plot_vec_title = "Campo Vetorial Composto f·F"

    # ==========================================================================
    # SEÇÃO 3: SUBSTITUIÇÃO NUMÉRICA E INTERPRETAÇÃO FÍSICA
    # ==========================================================================
    st.divider()
    st.subheader(f"3️⃣ Substituição Numérica no Ponto P({x0}, {y0}, {z0})")

    if res_type == "scalar":
        val_exact = res_sym.subs(subs_pt)
        val_float = float(val_exact.evalf())
        st.latex(rf"\text{{Resultado em }} P({x0}, {y0}, {z0}) = {sp.latex(val_exact)} \approx {val_float:.6f}")

        if "Divergente" in operador:
            if abs(val_float) < 1e-9:
                st.info("💡 **Interpretação Física no Ponto:** $\\nabla \\cdot \\vec{F} = 0$ — Comportamento **Solenoidal / Incompressível** (sem geração nem consumo líquido de fluxo no ponto).")
            elif val_float > 0:
                st.success(f"💡 **Interpretação Física no Ponto:** $\\nabla \\cdot \\vec{{F}} = {val_float:.4f} > 0$ — Presença de **Fonte (Source / Expansão)** no ponto avaliado.")
            else:
                st.warning(f"💡 **Interpretação Física no Ponto:** $\\nabla \\cdot \\vec{{F}} = {val_float:.4f} < 0$ — Presença de **Sorvedouro (Sink / Compressão)** no ponto avaliado.")
        elif "Laplaciano" in operador:
            if abs(val_float) < 1e-9:
                st.info("💡 **Interpretação Física no Ponto:** $\\nabla^2 f = 0$ — A função satisfaz a **Equação de Laplace (Função Harmônica / Regime Permanente sem geração)** no ponto.")
            else:
                st.info(f"💡 **Interpretação Física no Ponto:** $\\nabla^2 f = {val_float:.4f} \\neq 0$ — Comportamento governado pela **Equação de Poisson** (curvatura líquida não nula).")
    else:
        vx_ex = res_sym[0].subs(subs_pt)
        vy_ex = res_sym[1].subs(subs_pt)
        vz_ex = res_sym[2].subs(subs_pt)
        vx_fl, vy_fl, vz_fl = float(vx_ex.evalf()), float(vy_ex.evalf()), float(vz_ex.evalf())
        mag_fl = np.sqrt(vx_fl**2 + vy_fl**2 + vz_fl**2)

        st.latex(
            rf"\vec{{R}}({x0}, {y0}, {z0}) = \left({sp.latex(vx_ex)}\right)\hat{{i}} + "
            rf"\left({sp.latex(vy_ex)}\right)\hat{{j}} + \left({sp.latex(vz_ex)}\right)\hat{{k}} "
            rf"\approx ({vx_fl:.4f})\hat{{i}} + ({vy_fl:.4f})\hat{{j}} + ({vz_fl:.4f})\hat{{k}}"
        )
        st.latex(rf"\|\vec{{R}}({x0}, {y0}, {z0})\| = \sqrt{{({vx_fl:.4f})^2 + ({vy_fl:.4f})^2 + ({vz_fl:.4f})^2}} \approx {mag_fl:.6f}")

        if "Rotacional" in operador:
            if mag_fl < 1e-9:
                st.info("💡 **Interpretação Física no Ponto:** $\\nabla \\times \\vec{F} = \\vec{0}$ — Campo **Irrotacional / Conservativo** (ausência de vorticidade local).")
            else:
                st.success(f"💡 **Interpretação Física no Ponto:** $\\|\\nabla \\times \\vec{{F}}\\| = {mag_fl:.4f} > 0$ — Presença de **Vórtice / Redemoinho (Circulação / Vorticidade não nula)** ao redor do eixo dado pelo vetor resultante.")
        elif "Gradiente" in operador:
            st.info(f"💡 **Interpretação Física no Ponto:** O vetor $\\nabla f$ aponta na direção de **máxima taxa de crescimento** do potencial escalar, sendo ortogonal às curvas/superfícies equipotenciais. Em física, $\\vec{{E}} = -\\nabla V$ representa o campo de força/elétrico/gravitacional.")

# ==============================================================================
# ABA 2: REPRESENTAÇÃO GRÁFICA 2D (MATPLOTLIB)
# ==============================================================================
with tab_graf2d:
    st.subheader(f"Visualização 2D no Plano XY (Corte em z = {z_slice})")
    N2D = 25
    x_vals = np.linspace(-L_dom, L_dom, N2D)
    y_vals = np.linspace(-L_dom, L_dom, N2D)
    X2, Y2 = np.meshgrid(x_vals, y_vals)
    Z2 = np.full_like(X2, z_slice)

    S2 = eval_on_grid(plot_scalar_expr, X2, Y2, Z2)
    U2 = eval_on_grid(plot_vec_expr[0], X2, Y2, Z2)
    V2 = eval_on_grid(plot_vec_expr[1], X2, Y2, Z2)
    speed2 = np.sqrt(U2**2 + V2**2)

    fig2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

    # Subplot 1: Mapa de Contorno + Vetores (Quiver)
    if np.allclose(S2, S2[0, 0]):
        c1 = ax1.pcolormesh(X2, Y2, S2, shading='auto', cmap='viridis')
    else:
        c1 = ax1.contourf(X2, Y2, S2, levels=20, cmap='viridis', alpha=0.85)
    fig2d.colorbar(c1, ax=ax1, label="Intensidade Escalar")

    if not (np.allclose(U2, 0) and np.allclose(V2, 0)):
        ax1.quiver(X2, Y2, U2, V2, color='white', alpha=0.9)
    ax1.plot(x0, y0, 'ro', markersize=8, label=f'Ponto P({x0}, {y0})')
    ax1.set_title(f"{plot_scalar_title}\n+ Vetores", fontsize=10, fontweight='bold')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Subplot 2: Linhas de Corrente (Streamlines - Fonte/Sorvedouro/Vórtice)
    if not (np.allclose(U2, 0) and np.allclose(V2, 0)):
        strm = ax2.streamplot(X2, Y2, U2, V2, color=speed2, cmap='plasma', linewidth=1.5, density=1.2)
        fig2d.colorbar(strm.lines, ax=ax2, label="Magnitude Vetorial")
    else:
        ax2.text(0, 0, "Campo Vetorial Nulo no Plano", ha='center', va='center', fontsize=12)
    ax2.plot(x0, y0, 'ro', markersize=8, label=f'Ponto P({x0}, {y0})')
    ax2.set_title(f"Linhas de Corrente (Streamlines)\n{plot_vec_title}", fontsize=10, fontweight='bold')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(-L_dom, L_dom)
    ax2.set_ylim(-L_dom, L_dom)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2d)

# ==============================================================================
# ABA 3: REPRESENTAÇÃO GRÁFICA 3D INTERATIVA (PLOTLY)
# ==============================================================================
with tab_graf3d:
    st.subheader("Visualização 3D Interativa (Rotacione e aplique Zoom com o mouse)")
    col3d_1, col3d_2 = st.columns(2)

    # Gráfico 3D 1: Campo Vetorial 3D (Cones)
    with col3d_1:
        N3D = 6
        lin3 = np.linspace(-L_dom, L_dom, N3D)
        X3, Y3, Z3 = np.meshgrid(lin3, lin3, lin3)
        U3 = eval_on_grid(plot_vec_expr[0], X3, Y3, Z3)
        V3 = eval_on_grid(plot_vec_expr[1], X3, Y3, Z3)
        W3 = eval_on_grid(plot_vec_expr[2], X3, Y3, Z3)

        fig3d_vec = go.Figure()
        fig3d_vec.add_trace(go.Cone(
            x=X3.flatten(), y=Y3.flatten(), z=Z3.flatten(),
            u=U3.flatten(), v=V3.flatten(), w=W3.flatten(),
            colorscale='Viridis',
            sizemode='scaled',
            sizeref=0.4,
            name='Campo Vetorial'
        ))
        fig3d_vec.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode='markers',
            marker=dict(size=7, color='red'),
            name=f'Ponto P({x0},{y0},{z0})'
        ))
        fig3d_vec.update_layout(
            title=f"3D Vetorial: {plot_vec_title}",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            height=520,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d_vec, use_container_width=True)

    # Gráfico 3D 2: Superfície Escalar 3D (em z = z_slice)
    with col3d_2:
        fig3d_surf = go.Figure(data=[go.Surface(
            x=X2, y=Y2, z=S2,
            colorscale='Cividis',
            opacity=0.9,
            name='Superfície Escalar'
        )])
        val_p_surf = float(eval_on_grid(plot_scalar_expr, np.array([[x0]]), np.array([[y0]]), np.array([[z_slice]]))[0, 0])
        fig3d_surf.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[val_p_surf],
            mode='markers',
            marker=dict(size=7, color='red'),
            name=f'P({x0},{y0}) na Superfície'
        ))
        fig3d_surf.update_layout(
            title=f"Superfície 3D: {plot_scalar_title} (z = {z_slice})",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Valor Escalar'),
            height=520,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d_surf, use_container_width=True)