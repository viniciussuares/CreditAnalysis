/* Questions I will answer using public datasets on covid-19 in Brazil:

1. What has been the percentage of infected people that gets to occupy a hospital bed?
2. How does the lethality for people occupying a hospital bed compares to those in general affected by covid?
3. How has the occupation of hospital beds because of covid been varying over time?

* All of these questions will be answered by state or for the whole country.

*/


DROP SCHEMA covid CASCADE;


CREATE SCHEMA covid;

CREATE TABLE covid."beds_occupation"
   ( _id  CHAR(11) PRIMARY KEY
   , notification_date TIMESTAMP
   , cnes VARCHAR(50)
   , occup_suspects_clinic FLOAT
   , occup_suspects_ICU FLOAT
   , occup_confirmed_clinic FLOAT
   , occup_confirmed_ICU FLOAT
   , exit_suspects_death FLOAT
   , exit_suspects_discharge FLOAT
   , exit_confirmed_death FLOAT
   , exit_confirmed_discharge FLOAT
   , origin VARCHAR(60)
   , p_user VARCHAR(60)
   , notification_state VARCHAR(30)
   , notification_city VARCHAR(60)
   , _state VARCHAR(30)
   , city VARCHAR(60)
   , deleted BOOLEAN
   , validated BOOLEAN
   , created_at TIMESTAMP
   , updated_at TIMESTAMP );

-- The data was imported using the wizard
   
-- Changing notification_date and notification_state to synchronize all the visualizations later 

ALTER TABLE covid."beds_occupation"
ALTER COLUMN notification_date SET DATA TYPE DATE;

UPDATE covid."beds_occupation"
SET    notification_state = 'RS'
WHERE  notification_state = 'Rio Grande do Sul';

UPDATE covid."beds_occupation"
SET    notification_state = 'SC'
WHERE  notification_state = 'Santa Catarina';

UPDATE covid."beds_occupation"
SET    notification_state = 'PR'
WHERE  notification_state = 'Paraná';

UPDATE covid."beds_occupation"
SET    notification_state = 'SP'
WHERE  notification_state = 'São Paulo';

UPDATE covid."beds_occupation"
SET    notification_state = 'RJ'
WHERE  notification_state = 'Rio de Janeiro';

UPDATE covid."beds_occupation"
SET    notification_state = 'MG'
WHERE  notification_state = 'Minas Gerais';

UPDATE covid."beds_occupation"
SET    notification_state = 'ES'
WHERE  notification_state = 'Espírito Santo';

UPDATE covid."beds_occupation"
SET    notification_state = 'GO'
WHERE  notification_state IN ('Goiás', 'GOIAS');

UPDATE covid."beds_occupation"
SET    notification_state = 'DF'
WHERE  notification_state = 'Distrito Federal';

UPDATE covid."beds_occupation"
SET    notification_state = 'MT'
WHERE  notification_state = 'Mato Grosso';

UPDATE covid."beds_occupation"
SET    notification_state = 'MS'
WHERE  notification_state = 'Mato Grosso do Sul';

UPDATE covid."beds_occupation"
SET    notification_state = 'RN'
WHERE  notification_state = 'Rio Grande do Norte';

UPDATE covid."beds_occupation"
SET    notification_state = 'SE'
WHERE  notification_state = 'Sergipe';

UPDATE covid."beds_occupation"
SET    notification_state = 'PE'
WHERE  notification_state = 'Pernambuco';

UPDATE covid."beds_occupation"
SET    notification_state = 'AL'
WHERE  notification_state = 'Alagoas';

UPDATE covid."beds_occupation"
SET    notification_state = 'BA'
WHERE  notification_state = 'Bahia';

UPDATE covid."beds_occupation"
SET    notification_state = 'MA'
WHERE  notification_state = 'Maranhão';

UPDATE covid."beds_occupation"
SET    notification_state = 'CE'
WHERE  notification_state = 'Ceará';

UPDATE covid."beds_occupation"
SET    notification_state = 'PB'
WHERE  notification_state = 'Paraíba';

UPDATE covid."beds_occupation"
SET    notification_state = 'TO'
WHERE  notification_state = 'Tocantins';

UPDATE covid."beds_occupation"
SET    notification_state = 'AM'
WHERE  notification_state = 'Amazonas';

UPDATE covid."beds_occupation"
SET    notification_state = 'PI'
WHERE  notification_state = 'Piauí';

UPDATE covid."beds_occupation"
SET    notification_state = 'PA'
WHERE  notification_state = 'Pará';

UPDATE covid."beds_occupation"
SET    notification_state = 'RO'
WHERE  notification_state = 'Rondônia';

UPDATE covid."beds_occupation"
SET    notification_state = 'RR'
WHERE  notification_state = 'Roraima';

UPDATE covid."beds_occupation"
SET    notification_state = 'AC'
WHERE  notification_state = 'Acre';

UPDATE covid."beds_occupation"
SET    notification_state = 'AP'
WHERE  notification_state = 'Amapá';

-- Creating the view to be used in Power BI

CREATE OR REPLACE
   VIEW covid."hospital_beds" AS
   SELECT notification_date AS _date
   ,      notification_state AS _state
   ,      occup_confirmed_clinic
   +      occup_confirmed_ICU
   AS     occupation
   ,      exit_confirmed_death
   ,      exit_confirmed_discharge
   FROM   covid."beds_occupation"
   WHERE  deleted = 'False'
   AND    EXTRACT(YEAR FROM notification_date) IN (2020, 2021)
   AND    occup_confirmed_clinic
   +      occup_confirmed_ICU BETWEEN 0 AND 120000;
   
 -- The interval (0, 120000) leaves us with about 95% of the dataset. It disconsiders negative occupation and positive outliers.
   
CREATE TABLE covid."cases"
   ( region VARCHAR(20)
   , _state CHAR(2)
   , city VARCHAR(40)
   , state_code CHAR(2)
   , city_code CHAR(6)
   , region_code CHAR(5)
   , health_region_name VARCHAR(60)
   , _date DATE
   , epi_week INT
   , population INT
   , agg_cases INT
   , new_cases INT
   , agg_deaths INT
   , new_deaths INT
   , new_recovered INT
   , new_monitoring INT
   , countryside_metropolitan CHAR(1));
   
-- The data was imported using the wizard

SELECT * FROM covid."cases" LIMIT 10;

-- Creating the view

CREATE OR REPLACE
   VIEW covid."cases_brazil" AS
   SELECT   _date
   ,        _state
   ,       agg_cases
   +       new_cases
   AS      cases
   ,       agg_deaths
   +       new_deaths
   AS      deaths
   FROM    covid."cases"
   WHERE   agg_cases
   +       new_cases > 0;

-- Creating linking views to make relationships later

CREATE OR REPLACE
   VIEW   covid."linking_dates" AS
   SELECT DISTINCT _date
   FROM   covid."hospital_beds";