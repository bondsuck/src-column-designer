import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import re
import os
import urllib.request

# ==========================================
# 1. SYSTEM SETUP & STYLE
# ==========================================
st.set_page_config(page_title="Ultimate SRC Designer V3.0", page_icon="🏗️", layout="wide")

# Setup Font
@st.cache_resource
def setup_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    font_path = "Sarabun-Regular.ttf"
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve(font_url, font_path)
        except:
            return "sans-serif"
    try:
        fe = fm.FontEntry(fname=font_path, name='Sarabun')
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams['font.family'] = fe.name
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 11
        return fe.name
    except:
        return "sans-serif"

setup_font()

# CSS Styling
st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 400px !important; }
    .report-box { background-color: #f0f2f6; color: #000000 !important; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; font-family: monospace; white-space: pre-wrap; }
    div[data-testid="column"] { background-color: #ffffff; color: #000000 !important; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="column"] h3 { color: #000000 !important; }
    .stButton>button { width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENGINEERING DATA (UPDATED)
# ==========================================
# 1. Rebar Grade Database (ksc)
FY_GRADES = {
    "SR24": 2400,
    "SD30": 3000,
    "SD40": 4000,
    "SD50": 5000
}

# 2. Rebar Lists (Separated)
# Main Bar (Usually Deformed Bars)
main_rebar_list = ["DB12", "DB16", "DB20", "DB22", "DB25", "DB28", "DB32"]
# Stirrup Bar (Round + Small Deformed)
link_rebar_list = ["RB6", "RB9", "RB12", "DB10", "DB12", "DB16"]

# Combined DB for Area Calculation
rebar_db = {
    "RB6": 0.6, "RB9": 0.9, "RB12": 1.2,
    "DB10": 1.0, "DB12": 1.2, "DB16": 1.6, "DB20": 2.0, "DB22": 2.2,
    "DB25": 2.5, "DB28": 2.8, "DB32": 3.2
}

# H-Beam DB
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

def get_src_layers(D_conc, steel_key, custom_prop, axis='major'):
    prop = get_steel_prop(steel_key, custom_prop)
    if prop is None: return []
    d_s, bf_s = prop['d']/10.0, prop['bf']/10.0
    tw_s, tf_s = prop['tw']/10.0, prop['tf']/10.0
    layers = []
    if axis == 'major':
        gap = (D_conc - d_s) / 2.0
        layers.append({'A': bf_s*tf_s, 'd': gap + tf_s/2.0})
        layers.append({'A': (d_s - 2*tf_s)*tw_s, 'd': D_conc/2.0})
        layers.append({'A': bf_s*tf_s, 'd': D_conc - gap - tf_s/2.0})
    elif axis == 'minor':
        gap = (D_conc - bf_s) / 2.0
        total_area = (2 * bf_s * tf_s) + ((d_s - 2*tf_s) * tw_s)
        for i in range(5):
            layers.append({'A': total_area/5, 'd': gap + (bf_s/5/2.0) + i*(bf_s/5)})
    return layers

# ==========================================
# 3. CALCULATION LOGIC
# ==========================================

def parse_loads(raw_text, scale_seismic, mag_m2, mag_m3):
    if not raw_text: return []
    lines = raw_text.strip().split('\n'); processed_loads = []
    for i, line in enumerate(lines):
        line = line.replace(',', '').replace('\t', ' ')
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", line)]
        if len(nums) >= 5:
            P, M2, M3, V2, V3 = nums[0], nums[1], nums[2], nums[3], nums[4]
            processed_loads.append({
                'ID': f"L{i+1}", 'P': P * scale_seismic,
                'M2': M2 * scale_seismic * mag_m2, 'M3': M3 * scale_seismic * mag_m3,
                'V2': V2 * scale_seismic, 'V3': V3 * scale_seismic
            })
    return processed_loads

def calculate_shear_capacity_dynamic(W, D, fc, fy_stir, db_stir, s_stir, cover, db_main, steel_key, custom_prop, fy_steel, Nu_ton):
    phi_v = 0.75
    d_eff_x = W - cover - db_stir - db_main/2
    d_eff_y = D - cover - db_stir - db_main/2
    Av = 2 * (np.pi * db_stir**2 / 4)
    Ag = W * D
    Nu_kg = Nu_ton * 1000.0
    
    nu_factor = 1 + (Nu_kg / (140 * Ag)) if Nu_kg >= 0 else 1 + (Nu_kg / (35 * Ag))
    if Nu_kg >= 0: nu_factor = min(nu_factor, 3.5)
    else: nu_factor = max(nu_factor, 0)
        
    Vc_2 = 0.53 * np.sqrt(fc) * D * d_eff_x * nu_factor / 1000.0
    Vs_2 = (Av * fy_stir * d_eff_x / s_stir) / 1000.0
    Vc_3 = 0.53 * np.sqrt(fc) * W * d_eff_y * nu_factor / 1000.0
    Vs_3 = (Av * fy_stir * d_eff_y / s_stir) / 1000.0
    
    Vst_2 = 0.0; Vst_3 = 0.0
    prop = get_steel_prop(steel_key, custom_prop)
    if prop:
        d_cm, bf_cm = prop['d']/10.0, prop['bf']/10.0
        tw_cm, tf_cm = prop['tw']/10.0, prop['tf']/10.0
        Vst_3 = 0.6 * fy_steel * (d_cm * tw_cm) / 1000.0
        Vst_2 = 0.6 * fy_steel * (2 * bf_cm * tf_cm) / 1000.0

    PhiVn2 = phi_v * (Vc_2 + Vs_2 + Vst_2)
    PhiVn3 = phi_v * (Vc_3 + Vs_3 + Vst_3)
    return {'PhiVn2': PhiVn2, 'PhiVn3': PhiVn3, 'Vc2': Vc_2, 'Vs2': Vs_2, 'Vst2': Vst_2, 'Vc3': Vc_3, 'Vs3': Vs_3, 'Vst3': Vst_3, 'Nu_Factor': nu_factor}

def gen_pm_curve_src(width, depth, n_w, n_d, fc, fy_rebar, fy_steel, cover, db_main, steel_key, custom_prop, axis):
    As_b = np.pi * get_db(db_main)**2 / 4; bars = []
    for _ in range(n_w): bars.extend([{'A':As_b, 'd':cover}, {'A':As_b, 'd':depth-cover}])
    if n_d > 2:
        sp = (depth - 2*cover)/(n_d-1)
        for k in range(1, n_d-1): bars.extend([{'A':As_b, 'd':cover+k*sp}, {'A':As_b, 'd':cover+k*sp}])

    src_layers = get_src_layers(depth, steel_key, custom_prop, axis)
    c_vals = np.linspace(depth * 1.5, 0.1, 60)
    res_M, res_P = [], []
    beta1 = get_stress_block(fc)
    
    As_tot = len(bars)*As_b; Ast_tot = sum([l['A'] for l in src_layers])
    P0 = 0.85*fc*(width*depth - As_tot - Ast_tot) + As_tot*fy_rebar + Ast_tot*fy_steel
    Pn_max = (0.80 * 0.65 * P0) / 1000.0

    for c in c_vals:
        a = min(beta1*c, depth); Cc = 0.85*fc*a*width
        F_tot, M_tot = 0, 0; epsl = []; h_c = depth/2
        for br in bars:
            es = 0.003*(c-br['d'])/c; epsl.append(-es)
            fs = np.clip(es*Es, -fy_rebar, fy_rebar)
            F = br['A']*(fs - 0.85*fc) if (es > 0 and br['d'] < a) else br['A']*fs
            F_tot += F; M_tot += F*(h_c - br['d'])
        for st in src_layers:
            es = 0.003*(c-st['d'])/c; epsl.append(-es)
            fs = np.clip(es*Es, -fy_steel, fy_steel)
            F = st['A']*(fs - 0.85*fc) if (es > 0 and st['d'] < a) else st['A']*fs
            F_tot += F; M_tot += F*(h_c - st['d'])
        phi = get_phi_axial(max(epsl) if epsl else 0)
        res_P.append(phi*(Cc+F_tot)/1000); res_M.append(phi*(Cc*(h_c-a/2)+M_tot)/100000)
    return np.array(res_M), np.array(res_P), Pn_max

def interp_capacity(P_target, P_curve, M_curve):
    if P_target > np.max(P_curve): return 0.001
    return np.interp(P_target, P_curve[::-1], M_curve[::-1])

def process_loads(loads, Mn33, Pn33, Mn22, Pn22, Pmax_ton, section_data):
    processed = []
    W, D, fc, fy_stir, db_stir, s_stir, cover, db_main, steel_key, custom_prop, fy_steel = section_data
    for l in loads:
        Mox = max(0.1, interp_capacity(l['P'], Pn33, Mn33))
        Moy = max(0.1, interp_capacity(l['P'], Pn22, Mn22))
        rx, ry = abs(l['M3'])/Mox, abs(l['M2'])/Moy
        ur_pm = rx**1.5 + ry**1.5 
        shear_res = calculate_shear_capacity_dynamic(W, D, fc, fy_stir, db_stir, s_stir, cover, db_main, steel_key, custom_prop, fy_steel, l['P'])
        vn2, vn3 = shear_res['PhiVn2'], shear_res['PhiVn3']
        ur_v2 = abs(l['V2'])/vn2 if vn2 > 0 else 99
        ur_v3 = abs(l['V3'])/vn3 if vn3 > 0 else 99
        ur_shear = max(ur_v2, ur_v3)
        status = "PASS" if (ur_pm <= 1.0 and ur_shear <= 1.0 and l['P'] <= Pmax_ton) else "FAIL"
        processed.append({
            **l, 'UR_PM': ur_pm, 'UR_Shear': ur_shear, 'Status': status, 
            'Mox': Mox, 'Moy': Moy, 'RatioX': rx, 'RatioY': ry, 'shear_data': shear_res
        })
    return processed

def generate_step_text_src_v2(L, fy_stir_val, fy_main_val):
    shear = L['shear_data']
    reason = "Max Shear Ratio" if L['UR_Shear'] > L['UR_PM'] else "Max P-M Interaction Ratio"
    txt = f"CRITICAL CASE SELECTION: {L['ID']} (Reason: {reason})\n"
    txt += f"Design Loads (Factored): Pu={L['P']:.1f}T, Mu2={L['M2']:.1f}T-m, Mu3={L['M3']:.1f}T-m\n"
    txt += "="*60 + "\n"
    txt += f"PART A: AXIAL & MOMENT INTERACTION (Using Main Fy={fy_main_val} ksc)\n"
    txt += f"   • Capacity M33 (Major Axis): For Pu = {L['P']:.1f} T -> Phi*Mnx = {L['Mox']:.2f} T-m\n"
    txt += f"   • Capacity M22 (Minor Axis): For Pu = {L['P']:.1f} T -> Phi*Mny = {L['Moy']:.2f} T-m\n"
    txt += f"   • Ratio X = |Mu3|/PhiMnx = {abs(L['M3']):.2f}/{L['Mox']:.2f} = {L['RatioX']:.3f}\n"
    txt += f"   • Ratio Y = |Mu2|/PhiMny = {abs(L['M2']):.2f}/{L['Moy']:.2f} = {L['RatioY']:.3f}\n"
    txt += f"   • Interaction Equation: ({L['RatioX']:.3f})^1.5 + ({L['RatioY']:.3f})^1.5 = {L['UR_PM']:.3f}\n"
    txt += f"   • Status: {L['Status']} (Limit <= 1.0)\n\n"
    txt += f"PART B: SHEAR DESIGN (Using Stirrup Fy={fy_stir_val} ksc)\n"
    txt += f"   • Axial Load Effect (Nu Factor): {shear['Nu_Factor']:.3f} (Enhances Vc)\n"
    txt += f"   [Shear V2 - Minor Axis]\n"
    txt += f"     • Vc (Concrete) = {shear['Vc2']:.2f} T\n"
    txt += f"     • Vs (Stirrups) = {shear['Vs2']:.2f} T\n"
    txt += f"     • Vst (Steel Flanges) = {shear['Vst2']:.2f} T\n"
    txt += f"     • Phi*Vn2 = 0.75 * (Vc+Vs+Vst) = {shear['PhiVn2']:.2f} T\n"
    txt += f"     • Check: Vu2={abs(L['V2']):.1f} T / {shear['PhiVn2']:.2f} T -> Ratio={abs(L['V2'])/shear['PhiVn2']:.2f}\n"
    txt += f"   [Shear V3 - Major Axis]\n"
    txt += f"     • Vc (Concrete) = {shear['Vc3']:.2f} T\n"
    txt += f"     • Vs (Stirrups) = {shear['Vs3']:.2f} T\n"
    txt += f"     • Vst (Steel Web) = {shear['Vst3']:.2f} T\n"
    txt += f"     • Phi*Vn3 = 0.75 * (Vc+Vs+Vst) = {shear['PhiVn3']:.2f} T\n"
    txt += f"     • Check: Vu3={abs(L['V3']):.1f} T / {shear['PhiVn3']:.2f} T -> Ratio={abs(L['V3'])/shear['PhiVn3']:.2f}\n"
    return txt

# ==========================================
# 4. PLOTTING FUNCTION
# ==========================================
def plot_section_preview(W, D, cov, nx, ny, db_main, db_stir, steel_key, custom_prop, fc, fy_steel):
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.patch.set_facecolor('white')

    ax_img.add_patch(patches.Rectangle((0,0), W, D, ec='k', fc='#f5f5f5', lw=2))
    ax_img.add_patch(patches.Rectangle((cov,cov), W-2*cov, D-2*cov, ec='b', fc='none', ls='--', lw=1))
    
    prop = get_steel_prop(steel_key, custom_prop)
    Ast_steel = 0
    if prop:
        ds, bf, tw, tf = prop['d']/10.0, prop['bf']/10.0, prop['tw']/10.0, prop['tf']/10.0
        Ast_steel = (2 * bf * tf + (ds - 2*tf)*tw)
        cx, cy = W/2, D/2
        ax_img.add_patch(patches.Rectangle((cx-tw/2, cy-ds/2), tw, ds, fc='#444', ec='k'))
        ax_img.add_patch(patches.Rectangle((cx-bf/2, cy-ds/2), bf, tf, fc='#444', ec='k'))
        ax_img.add_patch(patches.Rectangle((cx-bf/2, cy+ds/2-tf), bf, tf, fc='#444', ec='k'))
        clearance_x = (W - bf)/2 - (cov + db_stir + db_main)
        status_txt = "OK" if clearance_x > 2.5 else "TIGHT"
        st_color = 'green' if clearance_x > 2.5 else 'red'
        sec_name = f"Custom {prop['d']:.0f}x{prop['bf']:.0f}" if steel_key=="Custom" else steel_key
    else:
        status_txt = "N/A"; clearance_x = 0; st_color='black'; sec_name = "None"

    coords = []
    for i in range(nx):
        sx = (W - 2*cov - 2*db_stir - db_main)/(nx-1) if nx>1 else 0
        x = cov+db_stir+db_main/2 + i*sx
        coords.extend([(x, D-cov-db_stir-db_main/2), (x, cov+db_stir+db_main/2)])
    if ny > 2:
        sy = (D - 2*cov - 2*db_stir - db_main)/(ny-1) if ny>1 else 0
        for j in range(1, ny-1):
            y = cov+db_stir+db_main/2 + j*sy
            coords.extend([(cov+db_stir+db_main/2, y), (W-cov-db_stir-db_main/2, y)])
    for x,y in coords: ax_img.add_patch(patches.Circle((x,y), db_main/2, color='#d62728', ec='k'))

    margin = max(W, D) * 0.15
    ax_img.set_xlim(-margin, W+margin); ax_img.set_ylim(-margin, D+margin)
    ax_img.axis('off'); ax_img.set_aspect('equal')
    ax_img.annotate('', xy=(-margin/4, 0), xytext=(-margin/4, D), arrowprops=dict(arrowstyle='<->', color='blue'))
    ax_img.text(-margin/4 - 1, D/2, f'h={D}', rotation=90, va='center', ha='right', color='blue', fontweight='bold')
    ax_img.annotate('', xy=(0, -margin/4), xytext=(W, -margin/4), arrowprops=dict(arrowstyle='<->', color='blue'))
    ax_img.text(W/2, -margin/4 - 1, f'b={W}', va='top', ha='center', color='blue', fontweight='bold')
    ax_img.set_title(f"Section {W}x{D} cm", fontweight='bold')

    ax_txt.axis('off')
    Ag = W*D; As_rebar = len(coords) * (np.pi*db_main**2/4)
    header = "SRC SECTION PROPERTIES"
    info_conc = f"[CONCRETE]\n  - Size: {W} x {D} cm\n  - Area (Ag): {Ag:.0f} cm2\n  - fc': {fc} ksc"
    info_steel = f"[STRUCTURAL STEEL]\n  - Section: {sec_name}\n  - Area (Ast): {Ast_steel:.2f} cm2\n  - Ratio: {Ast_steel/Ag*100:.2f} %\n  - Fy: {fy_steel} ksc"
    info_rebar = f"[REBAR]\n  - Main: {len(coords)}-DB{int(db_main*10)}\n  - Area (As): {As_rebar:.2f} cm2\n  - Ratio: {As_rebar/Ag*100:.2f} %"
    info_check = f"[CHECKS]\n  - Clearance (X): {clearance_x:.1f} cm  [{status_txt}]\n  - Total Steel: {(As_rebar+Ast_steel)/Ag*100:.2f} %"

    y = 1.0
    ax_txt.text(0, y, header, fontsize=12, fontweight='bold', color='#0056b3'); y-=0.15
    ax_txt.text(0, y, info_conc, fontsize=10, family='monospace', va='top'); y-=0.25
    ax_txt.text(0, y, info_steel, fontsize=10, family='monospace', va='top', color='#444'); y-=0.25
    ax_txt.text(0, y, info_rebar, fontsize=10, family='monospace', va='top', color='blue'); y-=0.25
    ax_txt.text(0, y, info_check, fontsize=10, family='monospace', va='top', bbox=dict(facecolor='#f0fff4' if status_txt=="OK" else '#fff5f5', edgecolor=st_color, pad=5))
    
    return fig

# ==========================================
# 5. STREAMLIT UI LAYOUT
# ==========================================
st.title("🏗️ Ultimate SRC Column Designer (V3.0)")
st.markdown("---")

# --- SIDEBAR INPUTS (Reorganized) ---
with st.sidebar:
    st.header("1️⃣ Section Properties")
    
    # 1. Concrete (Updated: Added Cover)
    with st.expander("Concrete & Section", expanded=True):
        col1, col2 = st.columns(2)
        w_b = col1.number_input("Width b (cm)", value=50.0, step=5.0)
        w_h = col2.number_input("Depth h (cm)", value=50.0, step=5.0)
        w_fc = col1.number_input("fc' (ksc)", value=280.0, step=10.0)
        w_cover = col2.number_input("Cover (cm)", value=4.0, step=0.5) # Moved here

    # 2. Structural Steel
    with st.expander("Structural Steel (SRC)", expanded=True):
        w_steel_key = st.selectbox("H-Beam Size", ["Custom"] + list(H_BEAM_STD.keys()), index=4)
        custom_prop = None
        if w_steel_key == "Custom":
            c1, c2 = st.columns(2)
            c_d = c1.number_input("d (mm)", value=300.0)
            c_bf = c2.number_input("bf (mm)", value=300.0)
            c_tw = c1.number_input("tw (mm)", value=10.0)
            c_tf = c2.number_input("tf (mm)", value=15.0)
            custom_prop = {'d': c_d, 'bf': c_bf, 'tw': c_tw, 'tf': c_tf}
            
        w_fy_steel = st.number_input("Fy Steel (ksc)", value=2400.0, step=100.0)

    # 3. Reinforcement (Updated: Grade Selection)
    with st.expander("Reinforcement", expanded=True):
        st.markdown("**Main Bars**")
        col1, col2 = st.columns(2)
        w_main_bar = col1.selectbox("Size", main_rebar_list, index=3)
        # Dropdown for Main Fy
        w_fy_main_key = col2.selectbox("Fy (Main)", ["SD30", "SD40", "SD50"], index=1)
        w_fy_main = FY_GRADES[w_fy_main_key] # Map to value
        
        c3, c4 = st.columns(2)
        w_nx = c3.number_input("Nx (side b)", value=3, min_value=2)
        w_ny = c4.number_input("Ny (side h)", value=3, min_value=2)
        
        st.markdown("---")
        st.markdown("**Stirrups**")
        col3, col4 = st.columns(2)
        w_stir_bar = col3.selectbox("Size ", link_rebar_list, index=1) # Space to differentiate key
        # Dropdown for Stirrup Fy
        w_fy_stir_key = col4.selectbox("Fy (Stirrup)", ["SR24", "SD30", "SD40", "SD50"], index=0)
        w_fy_stir = FY_GRADES[w_fy_stir_key] # Map to value
        
        w_stir_spacing = st.number_input("Spacing @ (cm)", value=15.0, step=1.0)

    st.header("2️⃣ Factors")
    w_seismic = st.number_input("Seismic Scale", value=1.0, step=0.1)
    w_d22 = st.number_input("Mag. Factor M22", value=1.0, step=0.1)
    w_d33 = st.number_input("Mag. Factor M33", value=1.0, step=0.1)

# --- MAIN AREA ---
col_main_L, col_main_R = st.columns([1, 1])

with col_main_L:
    st.subheader("🔍 Section Preview")
    db_m = get_db(w_main_bar)
    db_s = get_db(w_stir_bar)
    fig_sec = plot_section_preview(w_b, w_h, w_cover, w_nx, w_ny, db_m, db_s, w_steel_key, custom_prop, w_fc, w_fy_steel)
    st.pyplot(fig_sec)

with col_main_R:
    st.subheader("📝 Design Loads")
    st.info("Format: P  M2  M3  V2  V3 (Space separated)")
    default_input = "100 5 10 1 1\n300 10 20 2 2\n500 -15 30 5 5"
    w_input = st.text_area("Paste Data Here:", value=default_input, height=150)
    
    if st.button("🚀 Run Analysis", type="primary"):
        # --- Validation ---
        prop_check = get_steel_prop(w_steel_key, custom_prop)
        if prop_check:
            d_mm, bf_mm, tf_mm, tw_mm = prop_check['d'], prop_check['bf'], prop_check['tf'], prop_check['tw']
            if (d_mm/10.0 + 4 > w_h) or (bf_mm/10.0 + 4 > w_b):
                 st.error(f"⛔ **CRITICAL ERROR:** Steel section is too large for concrete! (Need min 2cm cover)"); st.stop()
            if w_steel_key == "Custom" and (2*tf_mm >= d_mm or tw_mm > bf_mm):
                 st.error(f"⛔ **PHYSICS ERROR:** Impossible geometry."); st.stop()

        with st.spinner("Calculating..."):
            section_data = (w_b, w_h, w_fc, w_fy_stir, db_s, w_stir_spacing, w_cover, db_m, w_steel_key, custom_prop, w_fy_steel)
            
            Mn33, Pn33, Pmax = gen_pm_curve_src(w_b, w_h, w_nx, w_ny, w_fc, w_fy_main, w_fy_steel, w_cover, db_m, w_steel_key, custom_prop, 'major')
            Mn22, Pn22, _ = gen_pm_curve_src(w_h, w_b, w_ny, w_nx, w_fc, w_fy_main, w_fy_steel, w_cover, db_m, w_steel_key, custom_prop, 'minor')
            
            loads = parse_loads(w_input, w_seismic, w_d22, w_d33)
            results = process_loads(loads, Mn33, Pn33, Mn22, Pn22, Pmax, section_data)
            
            st.session_state['results'] = results
            st.session_state['curves'] = (Mn33, Pn33, Mn22, Pn22, Pmax)
            st.session_state['fy_stir_used'] = w_fy_stir
            st.session_state['fy_main_used'] = w_fy_main

if 'results' in st.session_state:
    results = st.session_state['results']
    Mn33, Pn33, Mn22, Pn22, Pmax = st.session_state['curves']
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    st.info(f"ℹ️ Design Basis: Main Fy=**{st.session_state['fy_main_used']}**, Stirrup Fy=**{st.session_state['fy_stir_used']}** ksc")
    
    col_res1, col_res2 = st.columns([1.5, 1])
    
    with col_res1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(Mn33, Pn33, 'b', label='M33'); ax1.plot(Mn22, Pn22, 'g--', label='M22')
        ax1.plot(-Mn33, Pn33, 'b'); ax1.plot(-Mn22, Pn22, 'g--')
        ax1.axhline(Pmax, c='purple', ls=':', label=f'Pmax {Pmax:.0f}T'); ax1.legend(); ax1.grid(ls=':')
        ax1.set_title("Interaction Diagram"); ax1.set_xlabel("M (T-m)"); ax1.set_ylabel("P (T)")
        
        for l in results:
            c = 'g' if l['Status']=='PASS' else 'r'
            ax1.scatter(l['M3'], l['P'], c=c, s=30); ax1.scatter(l['M2'], l['P'], c=c, marker='x', s=30)
            
        t = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(t), np.sin(t), 'k-'); ax2.fill(np.cos(t), np.sin(t), 'g', alpha=0.1)
        for l in results:
            c = 'g' if l['Status']=='PASS' else 'r'
            ax2.scatter(l['M3']/l['Mox'] if l['Mox']>0 else 0, l['M2']/l['Moy'] if l['Moy']>0 else 0, c=c, s=50)
        ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_aspect('equal')
        ax2.set_title("Normalized Ratio Map")
        ax2.set_xlabel("Ratio M33 (Major)"); ax2.set_ylabel("Ratio M22 (Minor)"); ax2.grid(ls=':')
        st.pyplot(fig)
        
    with col_res2:
        st.subheader("📋 Summary Table")
        table_data = []
        for l in results:
            table_data.append({
                "Case": l['ID'], "Pu": f"{l['P']:.1f}", "Ratio PM": f"{l['UR_PM']:.3f}",
                "Ratio Shear": f"{l['UR_Shear']:.3f}", "Status": ("✅" if l['Status']=="PASS" else "❌") + " " + l['Status']
            })
        st.dataframe(table_data, hide_index=True)
        
    st.markdown("---")
    st.subheader("📝 Detailed Calculation (Critical Case)")
    if results:
        crit_case = max(results, key=lambda x: max(x['UR_PM'], x['UR_Shear']))
        calc_text = generate_step_text_src_v2(crit_case, st.session_state['fy_stir_used'], st.session_state['fy_main_used'])
        st.markdown(f'<div class="report-box">{calc_text}</div>', unsafe_allow_html=True)
