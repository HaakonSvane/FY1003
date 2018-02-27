from matplotlib import pyplot as plt
import numpy as np
import sympy as sp
import sys

show = True                                         # For debugging. Viser ikke grafer, men gjennomfører beregninger.
smoothing = 300                                     # Antall datapunkter for beregnet kurve

sp.init_printing(use_unicode=True)
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (16, 5), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
             'legend.handlelength': 1.5}
plt.rcParams.update(newparams)


# USIKKERHETER

deltaxb = 0.0005            # [m]
deltazb = 0.0005            # [m]
deltalb = 0.003             # [m]
deltaR_solb = 0.0005        # [m]
deltaR_spob = 0.0005        # [m]
deltaab = 0.001             # [m]
deltaIb = 0.01              # [A]

deltas = {"x":deltaxb, "z":deltazb, "l":deltalb, "R_sol":deltaR_solb, "R_spo":deltaR_spob, "a":deltaab, "I":deltaIb,}

theta_1, theta_2, a, mu0, I, R_sol, R_spo, N, l, x, z = sp.symbols('theta_1 theta_2 a mu0 I R_sol R_spo N l x z')


my_0 = 1.26e-06                 # Magnetisk permabillitet i tomt rom [H/m]
N_spo = 330                     # Antall viklinger kort spole
N_sol = 368                     # Antall viklinger solenoide
I_sys = 1.0                     # Spolestrøm [A]
r_spole = 0.07                  # Midlere spoleradius, kort [m]
r_solenoide = 0.05              # Indre radius, solenoide [m]
l_solenoide = 0.40              # Lengde solenoide [m]




formel_enkel_spole = ((((N * mu0 * I) / (R_spo * 2))) * (1 + (x ** 2) / R_spo ** 2) ** (-3 / 2))

formel_to_spoler = (((N * mu0 * I) / (R_spo * 2)) * (
(1 + ((x - a) ** 2) / (R_spo ** 2))**(-3/2) + (1 + ((x + a) ** 2) / (R_spo ** 2))**(-3/2)))

theta_1 = sp.acos((l / 2 + z) / (((l / 2 + z) ** 2 + R_sol ** 2) ** (1 / 2)))
theta_2 = sp.acos(-(l / 2 - z) / (((l / 2 - z) ** 2 + R_sol ** 2) ** (1 / 2)))
formel_solenoide = (N * mu0 * I) / (2 * l) * (sp.cos(theta_1) - sp.cos(theta_2))




###############################################################################################


# Forsøk nr. 1
x0_1 = 0.506 * 100

enSpol_data_fil = "EnSpole_Data.csv"
enSpol_data = np.loadtxt(enSpol_data_fil, delimiter=",")
x_1 = enSpol_data[:, 0]
B_1 = enSpol_data[:, 1]
z1_målt = []

for i in range(np.shape(x_1)[0]):
    z1_målt.append((x_1[i] - x0_1) / 100)

###############################################################################################


# Forsøk nr. 2
x0_2 = 0.488 * 100

toSpol_data_fil = "ToSpoler_Data.csv"
toSpoler_data = np.loadtxt(toSpol_data_fil, delimiter=",")
x_2 = toSpoler_data[:, 0]

B_2_1 = toSpoler_data[:, 1]
B_2_2 = toSpoler_data[:, 2]
B_2_3 = toSpoler_data[:, 3]
z2_målt = []

for i in range(np.shape(x_2)[0]):
    z2_målt.append((x_2[i] - x0_2) / 100)

###############################################################################################


# Forsøk nr. 3
x0_3 = 0.299 * 100

sol_data_fil = "Solenoide_Data.csv"
sol_data = np.loadtxt(sol_data_fil, delimiter=",")
x_3 = sol_data[:, 0]
B_3 = sol_data[:, 1]

z3_målt = []
for i in range(0, x_3.shape[0]):
    z3_målt.append((x_3[i] - x0_3) / 100)


###############################################################################################


# Analyse av feil mellom ressistans i instrument
enSpole_feil_data_fil = "EnSpole_feil_Data.csv"
enS_E = np.loadtxt(enSpole_feil_data_fil, delimiter=",")
B_E1 = enS_E[:]

    # For a = 1/2 * R
toSpoler_feil_data_fil = "ToSpoler_feil_Data.csv"
toS_E = np.loadtxt(toSpoler_feil_data_fil, delimiter=",")
B_E2 = toS_E[:]


###############################################################################################


def enSpole(points_to_calc = None):
    global x_s1
    f = formel_enkel_spole.subs([(mu0, my_0), (I, I_sys), (N, N_spo), (R_spo, r_spole)])
    x_s1 = np.linspace(z1_målt[0], z1_målt[-1], smoothing)
    B_calc_list_1 = []

    if points_to_calc is None:
        print("Kalkulerer B(x) for enkel spole...")
        for i in range(0, x_s1.shape[0]):

            sys.stdout.write('\r')
            sys.stdout.write("[%-19s] %d%%" % ('=' * int(i / len(x_s1) * 19 + 1), (1 + 100 * (i) / len(x_s1))))
            sys.stdout.flush()

            B_calc_list_1.append(f.subs([(x, x_s1[i])]))
    else:
        for n in points_to_calc:
            B_calc_list_1.append(f.subs([(x, n)]))
        print()

    return (np.array(B_calc_list_1, dtype=np.float64) * 1e4)


def toSpoler(dist_faktor, points_to_calc = None):
    global x_s2
    f = formel_to_spoler.subs([(mu0, my_0), (I, I_sys), (N, N_spo), (R_spo, r_spole), (a, 1/2*dist_faktor*r_spole)])
    x_s2 = np.linspace(z2_målt[0], z2_målt[-1], smoothing)
    B_calc_list_2 = []
    if points_to_calc is None:
        print("Kalkulerer B(x) for to spoler med " + str(dist_faktor) + "R avstand...")
        for i in range(0, x_s2.shape[0]):

            sys.stdout.write('\r')
            sys.stdout.write("[%-19s] %d%%" % ('=' * int(i / len(x_s2) * 19 + 1), (1 + 100 * (i) / len(x_s2))))
            sys.stdout.flush()

            B_calc_list_2.append(f.subs([(x, x_s2[i])]))
    else:
        for n in points_to_calc:
            B_calc_list_2.append(f.subs([(x, n)]))
        print()

    return (np.array(B_calc_list_2, dtype=np.float64) * 1e4)


def solenoide(points_to_calc = None):
    global x_s3
    f = formel_solenoide.subs([(mu0, my_0), (I, I_sys), (N, N_sol), (R_sol, r_solenoide), (l, l_solenoide)])
    x_s3 = np.linspace(z3_målt[0], z3_målt[-1], smoothing)
    B_calc_list_3 = []
    if points_to_calc is None:
        print("Kalkulerer B(x) for solenoide...")
        for i in range(0, x_s3.shape[0]):

            sys.stdout.write('\r')
            sys.stdout.write("[%-19s] %d%%" % ('=' * int(i / len(x_s3) * 19 + 1), (1 + 100 * (i) / len(x_s3))))
            sys.stdout.flush()

            B_calc_list_3.append(f.subs([(z, x_s3[i])]))
    else:
        for n in points_to_calc:
            B_calc_list_3.append(f.subs([(z,n)]))
        print()

    return (np.array(B_calc_list_3, dtype=np.float64) * 1e4)


def deltaF(formel, interval, name, dist_faktor = 0):

    """"" Generisk kalkulator for Gauss feilforplantningslov:
            Funksjonen tar inn en formel, partiellderiver returnerer en liste med de beregnede usikkerhetene for formelen
            gitt at alle variablene har blitt tilegnet verdi.
    """
    deltaF_list = np.zeros(np.shape(interval), dtype = np.float64)
    diff_list = []
    symb = list(formel.free_symbols)
    symb_list = []
    print("Partiellderiverer komponenter for beregning av "+name+" og beregner feil...")

    for i in range(0, len(symb)):
        if str(symb[i]) in deltas:
            diff_list.append(sp.diff(formel, symb[i]))
            symb_list.append(symb[i])
    for n in range(0, len(interval)):

        sys.stdout.write('\r')
        sys.stdout.write("[%-19s] %d%%" % ('=' * int(n/len(interval)*19+1), (1+ 100 * (n)/len(interval))))
        sys.stdout.flush()

        deltaF_list[n] = np.sum([(diff_list[d].subs([(x, interval[n]), (z,interval[n]), (mu0, my_0),(I, I_sys),(N, N_spo),(R_spo, r_spole),(R_sol, r_solenoide),(a, 1/2*dist_faktor*r_spole), (l, l_solenoide)])*deltas[str(symb_list[d])])**2 for d in range(len(symb_list))])
        deltaF_list[n] = np.sqrt(deltaF_list[n])*1e4
    print("\n")
    return deltaF_list




def visGrafe(z_smooth, B_beregnet, z_målt, B_målt, dB, B_på_gitt = []):
    deltaBe = np.around(B_målt * 0.004 + 0.001 * 100 + 0.01, decimals=2)

    a = plt.subplot(2,1,1)
    a.fill_between(z_smooth, B_beregnet - 1 * dB, B_beregnet + 1 * dB,
                     label='Beregnet kurve med $\pm \Delta B_\mathrm{b}$', alpha=0.5)

    a.errorbar(z_målt, B_målt, yerr=deltaBe , fmt="r", ecolor = "navy", label="Måledata", marker = "x", linestyle = "None",elinewidth=1)
    plt.ylim(0, 1.1 * np.max(B_beregnet + dB))
    plt.tick_params(axis = "x", top = "off", labelbottom = "off")
    en = plt.ylabel("Magnetfelt (Gauss)")

    plt.subplot(2,1,2, sharex = a)
    plt.fill_between(z_smooth, 100*(-1 * dB)/B_beregnet, 100*(1 * dB)/B_beregnet,
                     label='Beregnet kurve med $\pm \Delta B_\mathrm{b}$', alpha=0.5)
    plt.errorbar(z_målt, 100*(B_målt-B_på_gitt)/B_på_gitt, yerr=100*deltaBe/B_målt , fmt="r", ecolor = "navy", label="Måledata", elinewidth=1)
    for l in range(0,3):
        plt.axhline(y=l*2, linewidth=0.5, color="gray", alpha = 0.5)
        plt.axhline(y=-l*2, linewidth=0.5, color="gray", alpha = 0.5)

    plt.ylim(-1.3*np.max(100*(B_målt-B_på_gitt)/B_på_gitt), 1.3*np.max(100*(B_målt-B_på_gitt)/B_på_gitt))
    plt.xlabel("Avstand fra senter av spolen $[m]$")
    to = plt.ylabel("Relativt avvik $(\\%)$")
    plt.subplots_adjust(left=0.10, bottom=0.09, right=0.95, top=0.97,
                    wspace=0.20, hspace=0.05)
    plt.show()


p = 0
def visInstumentAvvik(z_målt, B_målt_E, B_målt, B_på_gitt):
    global p
    color = ""
    p += 1

    bG = B_på_gitt[:4]
    bG = np.append(bG,B_på_gitt[len(B_på_gitt)-4::])
    zM = z_målt[:4]
    zM = np.append(zM, z_målt[len(z_målt)-4::])
    bM = B_målt[:4]
    bM = np.append(bM, B_målt[len(B_målt)-4::])
    bME = B_målt_E

    if (p == 1): color = "blue"
    else: color = "green"

    c = plt.subplot(2, 2, int(p))
    c.plot(zM[:4], (bM[:4]-bG[:4]) / bG[:4] * 100, color = "orange")
    c.tick_params(axis = "y", left = "on", labelleft = "on")
    c.plot(zM[:4], (bME[:4] - bG[:4]) / bG[:4] * 100, color=color)
    plt.axhline(y=0, linewidth = 1, color = "gray")
    plt.axhline(y=3, linewidth=0.5, color="gray")
    plt.axhline(y=-3, linewidth=0.5, color="gray")
    if p == 1: plt.ylabel("Relativ differanse (\\%),\nen spole")
    else:
        plt.ylabel("Relativ differanse (\\%),\nto spoler ($a = 1/2*R$)")
        plt.xlabel("Avstand fra senter av spolen $[m]$")

    p += 1

    d = plt.subplot(2,2,int(p), sharey = c)
    d.plot(zM[4::], (bM[4::] - bG[4::]) / bG[4::] * 100, color="orange")
    d.plot(zM[4::],(bME[4::]-bG[4::])/bG[4::] * 100, color = color)
    d.axhline(y=0, linewidth = 1, color = "gray")
    plt.axhline(y=3, linewidth=0.5, color="gray")
    plt.axhline(y=-3, linewidth=0.5, color="gray")
    d.tick_params(axis = "y", left = "on", labelleft = "off")
    if p > 2: plt.xlabel("Avstand fra senter av spolen $[m]$")
    plt.ylim(-5.1, 5.1)

    if(p > 2):
        plt.show()


enSpole = [enSpole(), B_1, enSpole(z1_målt)]
dB_en = deltaF(formel_enkel_spole, x_s1, "en spole")

toSpoler_1 = [toSpoler(1 / 2), B_2_1, toSpoler(1 / 2, z2_målt)]
dB_to_1 = deltaF(formel_to_spoler, x_s2, "to spoler (a = R/2)", 1/2)

toSpoler_2 = [toSpoler(1), B_2_2, toSpoler(1, z2_målt)]
dB_to_2 = deltaF(formel_to_spoler, x_s2,"to spoler (a = R)", 1)

toSpoler_3 = [toSpoler(2), B_2_3, toSpoler(2, z2_målt)]
dB_to_3 = deltaF(formel_to_spoler, x_s2, "to spoler (a = 2R)", 1/2)

solenoide_c = [solenoide(), B_3, solenoide(z3_målt)]
dB_sol = deltaF(formel_solenoide, x_s3, "solenoide")

if show:
    visInstumentAvvik(z1_målt, B_E1, enSpole[1], enSpole[2])
    visInstumentAvvik(z2_målt, B_E2, toSpoler_1[1], toSpoler_1[2])

    visGrafe(x_s1, enSpole[0], z1_målt, enSpole[1], dB_en, enSpole[2])
    visGrafe(x_s2, toSpoler_1[0], z2_målt, toSpoler_1[1], dB_to_1, toSpoler_1[2])
    visGrafe(x_s2, toSpoler_2[0], z2_målt, toSpoler_2[1], dB_to_2, toSpoler_2[2])
    visGrafe(x_s2, toSpoler_3[0], z2_målt, toSpoler_3[1], dB_to_3, toSpoler_3[2])
    visGrafe(x_s3, solenoide_c[0], z3_målt, solenoide_c[1], dB_sol, solenoide_c[2])
