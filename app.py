import streamlit as st
import numpy as np
import matplotlib
# Force Backend ให้เป็น Agg เพื่อไม่ให้ตีกับ GUI ของ Server
matplotlib.use("Agg") 

# Import Figure class โดยตรง เพื่อเลี่ยง Global State ของ pyplot
from matplotlib.figure import Figure 
from matplotlib.backends.backend_agg import FigureCanvasAgg

import matplotlib.patches as patches
import matplotlib.font_manager as fm
import re
import os
import urllib.request
import gc # Garbage Collector เพื่อบังคับคืนแรม

# ==========================================
# 1. SYSTEM SETUP & STYLE (VERSION 3.6 Final)
# ==========================================
st.set_page_config(page_title="Ultimate SRC Designer v3.6 (Stable)", page_icon="🏗️", layout="wide")

@st.cache_resource
def setup_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    font_path = "Sarabun-Regular.ttf"
    if not os.path.exists(font_path):
        try: urllib.request.urlretrieve(font_url, font_path)
        except: return "sans-serif"
    try:
        fe = fm.FontEntry(fname=font_path, name='Sarabun')
        fm.fontManager.ttflist.insert(0, fe)
        matplotlib.rcParams['font.family'] = fe.name
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 11
        return fe.name
    except: return "sans-serif"

setup_font()

st.markdown("""
<style>
    .report-box { 
        background-color: rgba(128, 128, 128, 0.1); 
        color: inherit !important; 
        padding: 20px; 
        border-radius: 8px; 
        border-left: 5px solid #0d6efd; 
        font-family: monospace; 
        white-space: pre-wrap; 
        font-size: 14px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    div[data-testid="column"] { 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2); 
    }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENGINEERING DATA
# ==========================================
FY_GRADES = {"SR24": 2400, "SD30": 3000, "SD40": 4000, "SD50": 5000}
main_rebar_list = ["DB12", "DB16", "DB20", "DB22", "DB25", "DB28", "DB32"]
stirrup_rb_list = ["RB6", "RB9", "RB12"]
stirrup_db_list = ["DB10", "DB12", "DB16"]

# Database หน่วย cm
rebar_db = {
    "RB6": 0.6, "RB9": 0.9, "RB12": 1.2,
    "DB10": 1.0, "DB12": 1.2, "DB16": 1.6, "DB20": 2.0, "DB22": 2.2,
    "DB25": 2.5, "DB28": 2.8, "DB32": 3.2
}

# Database หน่วย mm (ต้องหาร 10 เพื่อเป็น cm ใน code)
H_BEAM_STD = {
    "None": None,
    "H-100x100": {'d': 100, 'bf': 100, 'tw': 6, 'tf': 8},
    "H-125x125": {'d': 125, 'bf': 125, 'tw': 6.5, 'tf': 9},
    "H-150x150": {'d': 150, 'bf': 150, 'tw': 7, 'tf': 10},
    "H-200x200": {'d': 200, 'bf': 200, 'tw': 8, 'tf': 12},
    "H-250x250": {'d': 250, 'bf': 250, 'tw': 9, 'tf': 14},
    "H-300x300": {'d': 300, 'bf': 300, 'tw': 10, 'tf': 15},
    "H-350x350": {'d': 350, 'bf': 350, 'tw': 12, 'tf': 19},
    "H-400x400": {'d': 400, 'bf': 400, 'tw': 13, 'tf': 21},
    "H-588x300": {'d': 588, 'bf': 300, 'tw': 12, 'tf': 20},
    "H-700x300": {'d': 700, 'bf': 300, 'tw': 13, 'tf': 24},
}
Es = 2040000

def get_db(n): return rebar_db.get(n, 0)
def get_stress_block(fc): return max(0.65, 0.85 - 0.05*(fc-280)/70) if fc > 280 else 0.85
def get_phi_axial(eps_t): return 0.65 if eps_t <= 0.002 else (0.90 if eps_t >= 0.005 else 0.65 + (eps_t - 0.002)*(250/3))
def get_steel_prop(key, custom_dict=None):
    if key == "Custom" and custom_dict: return custom_dict
    return H_BEAM_STD.get(key)

def get_src_layers(D_conc, steel_key, custom_prop, bending_axis='x'):
    prop = get_steel_prop(steel_key, custom_prop)
    if prop is None: return []
    # แปลง mm เป็น cm
    d_s, bf_s = prop['d']/10.0, prop['bf']/10.0
    tw_s, tf_s = prop['tw']/10.0, prop['tf']/10.0
    layers = []
    if bending_axis == 'x': 
        gap = (D_conc - d_s) / 2.0
        layers.append({'A': bf_s*tf_s, 'd': gap + tf_s/2.0})        
        layers.append({'A': (d_s - 2*tf_s)*tw_s, 'd': D_conc/2.0})  
        layers.append({'A': bf_s*tf_s, 'd': D_conc - gap - tf_s/2.0}) 
    elif bending_axis == 'y':
        gap = (D_conc - bf_s) / 2.0
        total_area = (2 * bf_s * tf_s) + ((d_s - 2*tf_s) * tw_s)
        for i in range(5):
            layers.append({'A': total_area/5, 'd': gap + (bf_s/5/2.0) + i*(bf_s/5)})
    return layers

# ==========================================
# 3. CALCULATION LOGIC
# ==========================================

def parse_loads(raw_text, scale_seismic, mag_mx, mag_my):
    if not raw_text: return []
    lines = raw_text.strip().split('\n'); processed_loads = []
    for i, line in enumerate(lines):
        line = line.replace(',', '').replace('\t', ' ')
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", line)]
        if len(nums) >= 5:
            processed_loads.append({
                'ID': f"L{i+1}", 
                'P': nums[0] * scale_seismic,
                'Mx': nums[1] * scale_seismic * mag_mx, 
                'My': nums[2] * scale_seismic * mag_my,
                'Vx': nums[3] * scale_seismic,
                'Vy': nums[4] * scale_seismic
            })
    return processed_loads

def calculate_shear_capacity_xy(W, D, fc, fy_stir, db_stir, s_stir, cover, db_main, steel_key, custom_prop, fy_steel, Nu_ton):
    phi_v = 0.75
    # Effective depth correction (subtract stirrup and half main bar)
    dist_center = cover + db_stir + db_main/2
    d_v_y = D - dist_center 
    d_v_x = W - dist_center 
    
    Av = 2 * (np.pi * db_stir**2 / 4)
    Ag = W * D
    Nu_kg = Nu_ton * 1000.0
    
    nu_factor = 1 + (Nu_kg / (140 * Ag)) if Nu_kg >= 0 else 1 + (Nu_kg / (35 * Ag))
    if Nu_kg >= 0: nu_factor = min(nu_factor, 3.5)
    else: nu_factor = max(nu_factor, 0)
        
    Vc_y = 0.53 * np.sqrt(fc) * W * d_v_y * nu_factor / 1000.0
    Vs_y = (Av * fy_stir * d_v_y / s_stir) / 1000.0
    
    Vc_x = 0.53 * np.sqrt(fc) * D * d_v_x * nu_factor / 1000.0
    Vs_x = (Av * fy_stir * d_v_x / s_stir) / 1000.0
    
    Vst_y = 0.0; Vst_x = 0.0
    prop = get_steel_prop(steel_key, custom_prop)
    if prop:
        d_cm, bf_cm = prop['d']/10.0, prop['bf']/10.0
        tw_cm, tf_cm = prop['tw']/10.0, prop['tf']/10.0
        Vst_y = 0.6 * fy_steel * (d_cm * tw_cm) / 1000.0
        Vst_x = 0.6 * fy_steel * (2 * bf_cm * tf_cm) / 1000.0

    return {
        'PhiVn_x': phi_v * (Vc_x + Vs_x + Vst_x), 
        'PhiVn_y': phi_v * (Vc_y + Vs_y + Vst_y),
        'Vc_x': Vc_x, 'Vs_x': Vs_x, 'Vst_x': Vst_x, 
        'Vc_y': Vc_y, 'Vs_y': Vs_y, 'Vst_y': Vst_y, 
        'Nu_Factor': nu_factor
    }

@st.cache_data
def gen_pm_curve_src(bending_dim, perp_dim, n_bend, n_perp, fc, fy_rebar, fy_steel, cover, db_main_val, db_stir_val, steel_key, custom_prop, axis_name):
    As_b = np.pi * db_main_val**2 / 4
    bars = []
    # Correct d_center: cover + stirrup + main/2
    d_center = cover + db_stir_val + db_main_val/2.0
    
    for _ in range(n_perp): 
        bars.extend([{'A':As_b, 'd':d_center}, {'A':As_b, 'd':bending_dim-d_center}])
    
    if n_bend > 2:
        sp = (bending_dim - 2*d_center)/(n_bend-1)
        for k in range(1, n_bend-1):
            bars.extend([{'A':As_b, 'd':d_center+k*sp}, {'A':As_b, 'd':d_center+k*sp}])

    src_layers = get_src_layers(bending_dim, steel_key, custom_prop, axis_name)
    
    c_vals = np.linspace(bending_dim * 1.5, 0.1, 60)
    res_M, res_P = [], []
    beta1 = get_stress_block(fc)
    
    As_tot = len(bars)*As_b; Ast_tot = sum([l['A'] for l in src_layers])
    P0 = 0.85*fc*(bending_dim*perp_dim - As_tot - Ast_tot) + As_tot*fy_rebar + Ast_tot*fy_steel
    Pn_max = (0.80 * 0.65 * P0) / 1000.0

    for c in c_vals:
        a = min(beta1*c, bending_dim)
        Cc = 0.85*fc*a*perp_dim
        F_tot, M_tot = 0, 0
        h_c = bending_dim/2
        epsl = []
        
        all_steel = bars + src_layers
        for st in all_steel:
            es = 0.003*(c-st['d'])/c
            epsl.append(-es)
            fy_curr = fy_rebar if st in bars else fy_steel
            fs = np.clip(es*Es, -fy_curr, fy_curr)
            if es > 0 and st['d'] < a:
                F = st['A']*(fs - 0.85*fc)
            else:
                F = st['A']*fs
            F_tot += F
            M_tot += F*(h_c - st['d'])
            
        phi = get_phi_axial(max(epsl) if epsl else 0)
        res_P.append(phi*(Cc+F_tot)/1000)
        res_M.append(phi*(Cc*(h_c-a/2)+M_tot)/100000)
        
    return np.array(res_M), np.array(res_P), Pn_max

def interp_capacity(P_target, P_curve, M_curve):
    if P_target > np.max(P_curve): return 0.001
    return np.interp(P_target, P_curve[::-1], M_curve[::-1])

def process_loads(loads, Mn_x, Pn_x, Mn_y, Pn_y, Pmax_ton, section_data):
    processed = []
    W, D, fc, fy_stir, db_stir, s_stir, cover, db_main, steel_key, custom_prop, fy_steel = section_data
    
    for l in loads:
        M_cap_x = max(0.1, interp_capacity(l['P'], Pn_x, Mn_x))
        M_cap_y = max(0.1, interp_capacity(l['P'], Pn_y, Mn_y))
        
        rx, ry = abs(l['Mx'])/M_cap_x, abs(l['My'])/M_cap_y
        ur_pm = rx**1.5 + ry**1.5 
        
        shear_res = calculate_shear_capacity_xy(W, D, fc, fy_stir, db_stir, s_stir, cover, db_main, steel_key, custom_prop, fy_steel, l['P'])
        vn_x, vn_y = shear_res['PhiVn_x'], shear_res['PhiVn_y']
        
        ur_vx = abs(l['Vx'])/vn_x if vn_x > 0 else 99
        ur_vy = abs(l['Vy'])/vn_y if vn_y > 0 else 99
        ur_shear = max(ur_vx, ur_vy)
        
        status = "PASS" if (ur_pm <= 1.0 and ur_shear <= 1.0 and l['P'] <= Pmax_ton) else "FAIL"
        
        processed.append({
            **l, 'UR_PM': ur_pm, 'UR_Shear': ur_shear, 'Status': status, 
            'M_cap_x': M_cap_x, 'M_cap_y': M_cap_y, 
            'Ratio_Mx': rx, 'Ratio_My': ry, 'shear_data': shear_res
        })
    return processed

def generate_step_text_src_xy(L, fy_stir_val, fy_main_val):
    shear = L['shear_data']
    reason = "Max Shear Ratio" if L['UR_Shear'] > L['UR_PM'] else "Max P-M Interaction Ratio"
    txt = f"CRITICAL CASE: {L['ID']} ({reason})\n"
    txt += f"Loads: Pu={L['P']:.1f}T, Mx={L['Mx']:.1f}T-m, My={L['My']:.1f}T-m, Vx={L['Vx']:.1f}T, Vy={L['Vy']:.1f}T\n"
    txt += "="*70 + "\n"
    txt += f"PART A: P-M INTERACTION (Main Fy={fy_main_val} ksc)\n"
    txt += f"  [Axis X - Moment around X]\n"
    txt += f"  • Capacity Phi*Mnx (at P={L['P']:.1f}) = {L['M_cap_x']:.2f} T-m\n"
    txt += f"  • Ratio X = |Mx|/PhiMnx = {L['Ratio_Mx']:.3f}\n"
    txt += f"  [Axis Y - Moment around Y]\n"
    txt += f"  • Capacity Phi*Mny (at P={L['P']:.1f}) = {L['M_cap_y']:.2f} T-m\n"
    txt += f"  • Ratio Y = |My|/PhiMny = {L['Ratio_My']:.3f}\n"
    txt += f"  • Check: ({L['Ratio_Mx']:.3f})^1.5 + ({L['Ratio_My']:.3f})^1.5 = {L['UR_PM']:.3f} [{L['Status']}]\n\n"
    txt += f"PART B: SHEAR DESIGN (Stirrup Fy={fy_stir_val} ksc)\n"
    txt += f"  • Axial Nu Factor: {shear['Nu_Factor']:.2f}\n"
    txt += f"  [Shear Vx - Horizontal]\n"
    txt += f"    • Phi*Vnx = {shear['PhiVn_x']:.2f} T (Vc={shear['Vc_x']:.1f}, Vs={shear['Vs_x']:.1f}, Vst={shear['Vst_x']:.1f})\n"
    txt += f"    • Ratio Vx = {abs(L['Vx'])/shear['PhiVn_x']:.3f}\n"
    txt += f"  [Shear Vy - Vertical]\n"
    txt += f"    • Phi*Vny = {shear['PhiVn_y']:.2f} T (Vc={shear['Vc_y']:.1f}, Vs={shear['Vs_y']:.1f}, Vst={shear['Vst_y']:.1f})\n"
    txt += f"    • Ratio Vy = {abs(L['Vy'])/shear['PhiVn_y']:.3f}\n"
    return txt

def plot_section_preview_xy(W, D, cov, nx, ny, db_main, db_stir, steel_key, custom_prop, fc, fy_steel):
    fig = Figure(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_img = fig.add_subplot(gs[0])
    ax_txt = fig.add_subplot(gs[1])
    
    ax_img.add_patch(patches.Rectangle((0,0), W, D, ec='k', fc='#f8f9fa', lw=2))
    ax_img.add_patch(patches.Rectangle((cov,cov), W-2*cov, D-2*cov, ec='b', fc='none', ls='--', lw=0.5))
    
    cx, cy = W/2, D/2
    margin = max(W, D) * 0.2
    ax_img.arrow(-margin, -margin/2, W+1.5*margin, 0, head_width=2, head_length=3, fc='r', ec='r', clip_on=False)
    ax_img.text(W+margin, -margin/2, 'X', color='red', fontweight='bold', fontsize=14, va='center')
    ax_img.arrow(-margin/2, -margin, 0, D+1.5*margin, head_width=2, head_length=3, fc='g', ec='g', clip_on=False)
    ax_img.text(-margin/2, D+margin, 'Y', color='green', fontweight='bold', fontsize=14, ha='center')

    prop = get_steel_prop(steel_key, custom_prop)
    Ast_steel = 0
    if prop:
        ds, bf, tw, tf = prop['d']/10.0, prop['bf']/10.0, prop['tw']/10.0, prop['tf']/10.0
        Ast_steel = (2 * bf * tf + (ds - 2*tf)*tw)
        ax_img.add_patch(patches.Rectangle((cx-tw/2, cy-ds/2), tw, ds, fc='#555', ec='k')) 
        ax_img.add_patch(patches.Rectangle((cx-bf/2, cy-ds/2), bf, tf, fc='#555', ec='k')) 
        ax_img.add_patch(patches.Rectangle((cx-bf/2, cy+ds/2-tf), bf, tf, fc='#555', ec='k')) 
        sec_name = f"Custom {prop['d']:.0f}x{prop['bf']:.0f}" if steel_key=="Custom" else steel_key
    else:
        sec_name = "None"

    coords = []
    if nx > 1:
        sx = (W - 2*cov - 2*db_stir - db_main)/(nx-1)
        for i in range(nx):
            x = cov+db_stir+db_main/2 + i*sx
            coords.extend([(x, D-cov-db_stir-db_main/2), (x, cov+db_stir+db_main/2)])
    if ny > 2:
        sy = (D - 2*cov - 2*db_stir - db_main)/(ny-1)
        for j in range(1, ny-1):
            y = cov+db_stir+db_main/2 + j*sy
            coords.extend([(cov+db_stir+db_main/2, y), (W-cov-db_stir-db_main/2, y)])
    coords = list(set(coords))
    for x,y in coords: ax_img.add_patch(patches.Circle((x,y), db_main/2, color='#d62728', ec='k'))

    ax_img.set_xlim(-margin, W+margin); ax_img.set_ylim(-margin, D+margin)
    ax_img.axis('off'); ax_img.set_aspect('equal')
    ax_img.text(W/2, D+2, f'b = {W} cm', ha='center', color='blue', fontweight='bold')
    ax_img.text(W+2, D/2, f'h = {D} cm', va='center', rotation=270, color='blue', fontweight='bold')

    ax_txt.axis('off')
    Ag = W*D; As_rebar = len(coords) * (np.pi*db_main**2/4)
    info = [("SRC PROPERTIES", "#0056b3", 12), (f"[CONCRETE] {W}x{D} cm", "black", 10), (f"[STEEL] {sec_name}", "#444", 10),
            (f"   Area: {Ast_steel:.2f} cm2 ({Ast_steel/Ag*100:.2f}%)", "#444", 10),
            (f"[REBAR] {len(coords)}-DB{int(db_main*10)}", "#d62728", 10),
            (f"   Area: {As_rebar:.2f} cm2 ({As_rebar/Ag*100:.2f}%)", "#d62728", 10),
            (f"[TOTAL STEEL] {(As_rebar+Ast_steel)/Ag*100:.2f} %", "green", 10)]
    y_pos = 1.0
    for txt, col, sz in info:
        ax_txt.text(0, y_pos, txt, fontsize=sz, color=col, fontweight='bold' if sz>10 else 'normal', family='monospace'); y_pos -= 0.12
    
    return fig

# ==========================================
# 4. UI LAYOUT (FIXED: Graph Position & Full Circle)
# ==========================================
st.title("🏗️ Ultimate SRC Designer v3.6 (Stable OO)")
st.markdown("---")

with st.sidebar:
    st.header("1️⃣ Section Config")
    with st.expander("Concrete", expanded=True):
        col1, col2 = st.columns(2)
        w_b = col1.number_input("Width b (X-dir)", value=50.0, step=5.0)
        w_h = col2.number_input("Depth h (Y-dir)", value=50.0, step=5.0)
        w_fc = col1.number_input("fc' (ksc)", value=280.0, step=10.0)
        w_cover = col2.number_input("Covering (cm)", value=3.0, step=0.5, help="Distance from surface to stirrup")

    with st.expander("Structural Steel", expanded=True):
        w_steel_key = st.selectbox("Size", ["Custom"] + list(H_BEAM_STD.keys()), index=4)
        custom_prop = None
        if w_steel_key == "Custom":
            c1, c2 = st.columns(2)
            c_d = c1.number_input("d (mm)", value=300.0); c_bf = c2.number_input("bf (mm)", value=300.0)
            c_tw = c1.number_input("tw (mm)", value=10.0); c_tf = c2.number_input("tf (mm)", value=15.0)
            custom_prop = {'d': c_d, 'bf': c_bf, 'tw': c_tw, 'tf': c_tf}
        w_fy_steel = st.number_input("Fy Steel (ksc)", value=2400.0)

    with st.expander("Rebars", expanded=True):
        col1, col2 = st.columns(2)
        w_main_bar = col1.selectbox("Main Bar", main_rebar_list, index=3)
        w_fy_main_key = col2.selectbox("Fy Main", ["SD30", "SD40", "SD50"], index=1)
        w_fy_main = FY_GRADES[w_fy_main_key]
        c3, c4 = st.columns(2)
        w_nx = c3.number_input("N bars (X-face)", value=3, min_value=2)
        w_ny = c4.number_input("N bars (Y-face)", value=3, min_value=2)
        st.markdown("---")
        col3, col4 = st.columns(2)
        w_fy_stir_key = col4.selectbox("Fy Stirrup", ["SR24", "SD30", "SD40"], index=0)
        w_fy_stir = FY_GRADES[w_fy_stir_key]
        cur_list = stirrup_rb_list if w_fy_stir_key == "SR24" else stirrup_db_list
        w_stir_bar = col3.selectbox("Stirrup Size", cur_list, index=1)
        w_stir_spacing = st.number_input("Spacing (cm)", value=15.0)

    st.header("2️⃣ Factors")
    w_seismic = st.number_input("Seismic Scale", value=1.0)
    w_mx_fac = st.number_input("Mag. Mx", value=1.0)
    w_my_fac = st.number_input("Mag. My", value=1.0)

# สร้าง Layout หลัก
col_L, col_R = st.columns([1.5, 1])

# ---------------------------------------------------------
# COLUMN LEFT: ส่วนแสดงผลรูปและกราฟ (ต้องเขียนในนี้ทั้งหมด)
# ---------------------------------------------------------
with col_L:
    st.subheader("🔍 Section Preview")
    db_m, db_s = get_db(w_main_bar), get_db(w_stir_bar)
    db_m_cm, db_s_cm = db_m, db_s
    
    # 1. วาดรูป Section
    fig_sec = plot_section_preview_xy(w_b, w_h, w_cover, w_nx, w_ny, db_m, db_s, w_steel_key, custom_prop, w_fc, w_fy_steel)
    st.pyplot(fig_sec)
    del fig_sec
    gc.collect()

    # 2. วาดกราฟ P-M (เช็ค session_state แล้ววาด "ข้างใน" col_L นี้เลย)
    if 'results' in st.session_state:
        res = st.session_state['results']
        Mnx, Pnx, Mny, Pny, Pmax = st.session_state['curves']
        
        st.markdown("---")
        st.subheader("📈 P-M Interaction Diagram")
        
        # ปรับขนาด Figure ให้สูงขึ้น
        fig = Figure(figsize=(10, 6.5), dpi=100)
        fig.patch.set_facecolor('white')
        
        # Grid: ซ้าย(P-M) ขวา(Ratio)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # --- [GRAHP 1] P-M Diagram ---
        ax1.plot(Mnx, Pnx, 'r-', label='Mx Cap')
        ax1.plot(Mny, Pny, 'b--', label='My Cap')
        ax1.plot(-Mnx, Pnx, 'r-')
        ax1.plot(-Mny, Pny, 'b--')
        ax1.axhline(Pmax, c='k', ls=':', label='Pmax')
        
        # Plot จุด Load Cases
        for r in res:
            col = 'g' if r['Status']=='PASS' else 'r'
            ax1.scatter(abs(r['Mx']), r['P'], c=col, marker='o', s=40, zorder=5, label='_nolegend_')
            ax1.scatter(abs(r['My']), r['P'], c=col, marker='x', s=40, zorder=5, label='_nolegend_')

        # จัดแกน Y ให้ครอบคลุมทั้งแรงอัดและแรงดึง
        y_all = np.concatenate([Pnx, Pny])
        y_min, y_max = np.min(y_all), np.max(y_all)
        ax1.set_ylim(y_min * 1.2, y_max * 1.2) # เผื่อขอบ 20%
        
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(ls=':', alpha=0.6)
        ax1.set_xlabel('Moment Strength (T-m)', fontsize=10)
        ax1.set_ylabel('Axial Strength (T)', fontsize=10)
        ax1.set_title("P-M Capacity Check", fontweight='bold', fontsize=11)

        # --- [GRAPH 2] Interaction Ratio (Full Circle) ---
        # วาดวงกลมเต็มวง
        theta = np.linspace(0, 2*np.pi, 100)
        x_circ = np.cos(theta)
        y_circ = np.sin(theta)
        
        ax2.plot(x_circ, y_circ, 'k-', lw=1.5)
        ax2.fill(x_circ, y_circ, '#d4edda', alpha=0.5) # สีเขียวอ่อนพื้นหลัง
        
        # เส้นเล็งกึ่งกลาง
        ax2.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.axvline(0, color='gray', lw=0.5, ls='--')

        # Plot จุด Ratio
        for r in res:
            col = 'g' if r['Status']=='PASS' else 'r'
            # จุด
            ax2.scatter(r['Ratio_Mx'], r['Ratio_My'], c=col, s=80, edgecolors='k', zorder=10)
            # ป้ายชื่อ ID
            ax2.text(r['Ratio_Mx']+0.05, r['Ratio_My']+0.05, r['ID'], fontsize=9, color='blue', fontweight='bold')

        # [FIX] Set Limit ให้เห็นเต็มวง (-1.5 ถึง 1.5)
        limit_val = 1.5
        ax2.set_xlim(-limit_val, limit_val)
        ax2.set_ylim(-limit_val, limit_val)
        ax2.set_aspect('equal') # บังคับสัดส่วน 1:1 เป็นวงกลม

        # [FIX] ใส่ชื่อแกน X, Y
        ax2.set_xlabel(r'Ratio X ($\frac{M_{ux}}{\phi M_{nx}}$)', fontsize=10)
        ax2.set_ylabel(r'Ratio Y ($\frac{M_{uy}}{\phi M_{ny}}$)', fontsize=10)
        ax2.set_title("Interaction Ratio (Comb.)", fontweight='bold', fontsize=11)
        ax2.grid(True, ls=':', alpha=0.5)

        fig.tight_layout()
        st.pyplot(fig)
        
        del fig
        gc.collect()

# ---------------------------------------------------------
# COLUMN RIGHT: Input Loads
# ---------------------------------------------------------
with col_R:
    st.subheader("📋 Input Loads")
    st.info("Format: P  Mx  My  Vx  Vy")
    default_input = "100 5 10 2 3\n300 20 15 5 8\n500 -10 30 8 5"
    w_input = st.text_area("Paste loads here:", value=default_input, height=200)
    
    if st.button("🚀 Calculate Check", type="primary"):
        with st.spinner("Analyzing..."):
            sec_data = (w_b, w_h, w_fc, w_fy_stir, db_s_cm, w_stir_spacing, w_cover, db_m_cm, w_steel_key, custom_prop, w_fy_steel)
            
            # 1. Generate Curves
            Mn_x, Pn_x, Pmax = gen_pm_curve_src(w_h, w_b, w_ny, w_nx, w_fc, w_fy_main, w_fy_steel, w_cover, db_m_cm, db_s_cm, w_steel_key, custom_prop, 'x')
            Mn_y, Pn_y, _ = gen_pm_curve_src(w_b, w_h, w_nx, w_ny, w_fc, w_fy_main, w_fy_steel, w_cover, db_m_cm, db_s_cm, w_steel_key, custom_prop, 'y')
            
            # 2. Process Loads
            loads = parse_loads(w_input, w_seismic, w_mx_fac, w_my_fac)
            results = process_loads(loads, Mn_x, Pn_x, Mn_y, Pn_y, Pmax, sec_data)
            
            # 3. Save to Session
            st.session_state['results'] = results
            st.session_state['curves'] = (Mn_x, Pn_x, Mn_y, Pn_y, Pmax)
            st.session_state['materials'] = (w_fy_stir, w_fy_main)
            
            # [IMPORTANT] Rerun เพื่อบังคับให้ Code ใน col_L ทำงานใหม่และแสดงกราฟทันที
            st.rerun()

# ---------------------------------------------------------
# FOOTER: ตารางสรุป (อยู่นอก Column L/R เพื่อให้กว้างเต็มจอ)
# ---------------------------------------------------------
if 'results' in st.session_state:
    res = st.session_state['results']
    fy_s, fy_m = st.session_state['materials']

    st.markdown("---")
    st.header("📝 Design Summary")
    
    c_sum1, c_sum2 = st.columns([1, 1.5])
    
    with c_sum1:
        st.caption("Load Case Status")
        t_data = [{"ID":r['ID'], "Pu":f"{r['P']:.0f}", "PM Ratio":f"{r['UR_PM']:.2f}", "Shear Ratio":f"{r['UR_Shear']:.2f}", "Status":r['Status']} for r in res]
        st.dataframe(t_data, hide_index=True, use_container_width=True)

    with c_sum2:
        if res:
            st.caption("Critical Calculation Step")
            crit = max(res, key=lambda x: max(x['UR_PM'], x['UR_Shear']))
            st.markdown(f'<div class="report-box">{generate_step_text_src_xy(crit, fy_s, fy_m)}</div>', unsafe_allow_html=True)
