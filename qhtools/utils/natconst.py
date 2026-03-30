"""Natural constants in CGS units.

Translated from RADMC's IDL function ``problem_natconst.pro``, with
additional astrophysical quantities appended at the end.

Fundamental Constants
---------------------
====    =======================================
Name    Description
====    =======================================
gg      Gravitational constant [cm^3/g/s^2]
mp      Proton mass [g]
me      Electron mass [g]
kk      Boltzmann's constant [erg/K]
hh      Planck's constant [erg s]
ee      Elementary charge [esu]
cc      Speed of light [cm/s]
st      Thomson cross-section [cm^2]
ss      Stefan-Boltzmann constant [erg/cm^2/K^4/s]
aa      Radiation constant, 4 ss / cc [erg/cm^3/K^4]
====    =======================================

Gas Constants
-------------
====    =======================================
muh2    Mean molecular weight (H2 + He + metals)
====    =======================================

Unit Conversions
----------------
====    =======================================
ev      Electronvolt [erg]
kev     Kilo-electronvolt [erg]
micr    Micron [cm]
km      Kilometre [cm]
angs    Angstrom [cm]
====    =======================================

Solar & Astronomical Constants
------------------------------
====    =======================================
ls      Solar luminosity [erg/s]
rs      Solar radius [cm]
ms      Solar mass [g]
ts      Solar effective temperature [K]
au      Astronomical unit [cm]
pc      Parsec [cm]
====    =======================================

Planetary Constants
-------------------
====    =======================================
mea     Earth mass [g]
rea     Earth equatorial radius [cm]
mmo     Moon mass [g]
rmo     Moon radius [cm]
dmo     Earth–Moon distance (centre-to-centre) [cm]
mju     Jupiter mass [g]
rju     Jupiter equatorial radius [cm]
dju     Jupiter–Sun distance [cm]
====    =======================================

Time Units
----------
====    =======================================
year    Year [s]
hour    Hour [s]
day     Day [s]
====    =======================================

Derived / Custom Quantities
---------------------------
==========  =======================================
Name        Description
==========  =======================================
gamma       Adiabatic index (5/3 for monoatomic ideal gas)
mus         Mean molecular weight of the Sun
knorm       Normalised Boltzmann factor, kk / (mus * mp) [erg/g/K]
A_C         Carbon mass fraction at solar metallicity
A_H         Hydrogen mass fraction at solar metallicity
sigma_t     Thomson cross-section [cm^2] (high-precision value)
log_csi     log10 of the Eddington luminosity-to-solar-luminosity ratio
==========  =======================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------
gg = 6.672e-8          # Gravitational constant [cm^3 g^-1 s^-2]
mp = 1.6726e-24        # Proton mass [g]
me = 9.1095e-28        # Electron mass [g]
kk = 1.3807e-16        # Boltzmann's constant [erg K^-1]
hh = 6.6262e-27        # Planck's constant [erg s]
ee = 4.8032e-10        # Elementary charge [esu]
cc = 2.99792458e10     # Speed of light [cm s^-1]
st = 6.6524e-25        # Thomson cross-section [cm^2]
ss = 5.6703e-5         # Stefan-Boltzmann constant [erg cm^-2 K^-4 s^-1]
aa = 7.5657e-15        # Radiation constant, 4 ss / cc [erg cm^-3 K^-4]

# ---------------------------------------------------------------------------
# Gas constants
# ---------------------------------------------------------------------------
muh2 = 2.3000e0        # Mean molecular weight (H2 + He + metals)

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
ev   = 1.6022e-12      # Electronvolt [erg]
kev  = 1.6022e-9       # Kilo-electronvolt [erg]
micr = 1e-4            # Micron [cm]
km   = 1e5             # Kilometre [cm]
angs = 1e-8            # Angstrom [cm]

# ---------------------------------------------------------------------------
# Solar & astronomical constants
# ---------------------------------------------------------------------------
ls  = 3.8525e33        # Solar luminosity [erg s^-1]
rs  = 6.96e10          # Solar radius [cm]
ms  = 1.99e33          # Solar mass [g]
ts  = 5.780e3          # Solar effective temperature [K]
au  = 1.496e13         # Astronomical unit [cm]
pc  = 3.08572e18       # Parsec [cm]

# ---------------------------------------------------------------------------
# Planetary constants
# ---------------------------------------------------------------------------
mea = 5.9736e27        # Earth mass [g]
rea = 6.375e8          # Earth equatorial radius [cm]
mmo = 7.347e25         # Moon mass [g]
rmo = 1.738e8          # Moon radius [cm]
dmo = 3.844e10         # Earth–Moon distance (centre-to-centre) [cm]
mju = 1.899e30         # Jupiter mass [g]
rju = 7.1492e9         # Jupiter equatorial radius [cm]
dju = 7.78412e13       # Jupiter–Sun distance [cm]

# ---------------------------------------------------------------------------
# Time units
# ---------------------------------------------------------------------------
year = 3.1536e7        # Year [s]
hour = 3.6000e3        # Hour [s]
day  = 8.6400e4        # Day [s]

# ---------------------------------------------------------------------------
# Derived / custom quantities
# ---------------------------------------------------------------------------
gamma   = 5.0 / 3.0                # Adiabatic index (monoatomic ideal gas)
mus     = 0.61                     # Mean molecular weight of the Sun
knorm   = kk / (mus * mp)          # Normalised Boltzmann factor [erg g^-1 K^-1]
A_C     = 2.69e-4                  # Carbon mass fraction at solar metallicity
A_H     = 0.76                     # Hydrogen mass fraction at solar metallicity

# Eddington ratio: log10(L_Edd / L_sun)
sigma_t = 6.6524587e-25            # Thomson cross-section [cm^2] (high precision)
log_csi = np.log10(4 * np.pi * gg * ms * mp * cc / sigma_t / ls)
