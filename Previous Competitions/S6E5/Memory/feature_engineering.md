# S6E5 Feature Engineering Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first)
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement | 🔬 Research

### 📝 Feature Entry Format
| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|
| `Drop_2023_Data` | Purging Year==2023 | N/A | **-0.02497 LB** | ❌ Removed |
| `Config D` | `TyreLife_sq`, `Degradation_Rate`, `RPxTL`, `Compound_Stint_` | TBD | **+0.00004 LB** | ✅ Winning |
| `TE_Row_Stats`| mean, std, min, max, range of TE features | TBD | **+0.00040 LB** | ✅ Winning |
| `RealMLP_Arch` | PyTabKit RealMLP | N/A | **Base architecture** | ✅ Used |
| `Race_Compound_TE` | Target Encoding on (Race x Compound) | TBD | Strong categorical signal | ✅ Used |
| `Race_Year_TE` | Target Encoding on (Race x Year) | TBD | Track-season signal | ✅ Used |
| `RaceProgress_200` | `pd.qcut(RaceProgress, 200)` | TBD | High-res phase signal | ✅ Used |
| `TyreLife_Bin_10` | `pd.qcut(TyreLife, 10)` | TBD | Non-linear signal | ✅ Used |
| `LapNumber_RP_Ratio` | `LapNumber / RaceProgress` | TBD | Normalizing race progression | ✅ Used |
| `TyreLife_Lap_Ratio` | `TyreLife / LapNumber` | TBD | Normalizing tire wear | ✅ Used |
| `Count_Encode` | Driver, Compound, Race, Year, PitStop counts | TBD | Frequency signals | ✅ Used |
| `Orig_Data_Concat` | Per-fold concatenation of original data | N/A | Data augmentation | ✅ Used |
| `Bigrams` | (Feature A, Feature B) combos | N/A | High redundancy/noise | ❌ Removed |
| `Digit_FE` | Extraction of digits from float strings | N/A | Synthetic noise artifact | ❌ Removed |
| `Num_to_Cat` | Converting all floats to cats | N/A | Loss of precision | ❌ Removed |
| `Drop_RaceProgress` | Drop due to r=0.96 with `LapNumber` | TBD | Redundancy reduction | ⚠️ Research |

---