This tutorial includes all code examples from Faleh, R., Morelli, S., Andriamiarana, V., Zachary, R. J., Flückiger, C., Brandt, H. https://arxiv.org/abs/2508.12983, Evaluating Psychometric test properties in intensive longitudinal data using dynamic structural equation models.

### Data

The data originate from Flückiger, C., Forrer, L., Schnider, B., Bättig, I., Bodenmann, G., & Zinbarg, R. E. (2016). A single-blinded, randomized clinical trial of how to implement an evidence-based treatment for generalized anxiety disorder [IMPLEMENT] — effects of three different strategies of implementation. EBioMedicine, 3, 163-171. doi: https://doi.org/10.1016/j.ebiom.2015.11.049 .

They include N = 57 participants who received Nt = 14 sessions of cognitive behavioral therapy. Before each session, participants completed the Beck Anxiety Inventory (BAI), and after each session, they completed the Working Alliance Inventory (WAI). To model the latent construct of anxiety, we use 3 parcels derived from the BAI. For the therapeutic alliance, we include 9 WAI items, with 3 items each representing the subscales: task, goals, and bond.

### Repository Structure

The repository is organized as follows:

#### 01_cfa
- `illustration01`: A four-factor CFA model at two time points (t = 1 and t = 15).

#### 02_arma
- `illustration02`: An AR(1) model for a single individual (i = 8), applied to one observed variable (y1) and to a latent factor indicated by the first three items (y1–y3).  
  Additional time structures (AR(2) and ARMA(1,1)) are provided in the appendix.

#### 03_mlm
- `illustration03`: A two-level AR(1) model for all individuals and time points, using a single observed variable (y1) with person-specific random intercept and slope.  
  A variant with an additional time-specific random intercept is also included.  
  A two-level ARMA(1,1) model is provided in the appendix.

#### 04_dsem
- `illustration04`: A two-level DSEM model for all individuals and time points, using a single latent factor (loading on items y1–y3) with an AR(1) time structure and person-specific random intercept and slope and a variant with an additional time-specific random intercept.  
  An extension to a four-factor model is included in the appendix.

#### 05_dlcsem
- `illustration05`: A one-factor state-switching DLCSEM with state-dependent AR(1) parameters including a person-specific random intercept and slope, demonstrating dynamic switching between high and low anxiety states.  
  *Note: May take several minutes to run.*

#### 06_fusion
- `illustration06`: A three-factor state-switching model illustrating the collapse of the working alliance subscales (task, goals, bond) into a single factor.
  *Note: May take several hours to run.*

All models are specified in both **JAGS** and **Stan**. Model files can be found in the `models` subfolder within each chapter directory.
