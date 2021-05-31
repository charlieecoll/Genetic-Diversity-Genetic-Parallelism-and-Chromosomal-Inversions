##################################################################
##                   PCA PARALLELISM CLUSTERS                   ##
##################################################################

#### Prepare the data ####

# Load the required libraries
library(ggplot2)
library(MASS)
library(dplyr)
library(tidyverse)
library(caret)
library(klaR)
library(ggpubr)
library(ggrepel)
library(reshape2)
library(ggdendro)
library(grid)
library(ggridges)

# Set working directory
setwd("<Working_Directory>")

# Import PCA dataframes
pca5 = read.table("<Input_File>", sep="\t")
pca6 = read.table("<Input_File>", sep="\t")
pca10 = read.table("<Input_File>", sep="\t")
pca11 = read.table("<Input_File>", sep="\t")
pca12 = read.table("<Input_File>", sep="\t")
pca13 = read.table("<Input_File>", sep="\t")
pca16 = read.table("<Input_File>", sep="\t")
pca18 = read.table("<Input_File>", sep="\t")
pca20 = read.table("<Input_File>", sep="\t")
pca22 = read.table("<Input_File>", sep="\t")
pca25 = read.table("<Input_File>", sep="\t")
pca27 = read.table("<Input_File>", sep="\t")

# Transform type into a factor
pca5$Type = as.factor(pca5$Type)
pca6$Type = as.factor(pca6$Type)
pca10$Type = as.factor(pca10$Type)
pca11$Type = as.factor(pca11$Type)
pca12$Type = as.factor(pca12$Type)
pca13$Type = as.factor(pca13$Type)
pca16$Type = as.factor(pca16$Type)
pca18$Type = as.factor(pca18$Type)
pca20$Type = as.factor(pca20$Type)
pca22$Type = as.factor(pca22$Type)
pca25$Type = as.factor(pca25$Type)
pca27$Type = as.factor(pca27$Type)

# Create the training and test data
train_pca5 = pca5[which((pca5$Concrete_Location=="Alaska" & pca5$Type=="Freshwater") | (pca5$Concrete_Location=="Alaska" | pca5$Concrete_Location=="East_Russia") & pca5$Type=="Marine"),]
test_pca5 = anti_join(pca5, train_pca5)
train_pca6 = pca6[which((pca6$Concrete_Location=="Alaska" & pca6$Type=="Freshwater") | (pca6$Concrete_Location=="Alaska" | pca6$Concrete_Location=="East_Russia") & pca6$Type=="Marine"),]
test_pca6 = anti_join(pca6, train_pca6)
train_pca10 = pca10[which((pca10$Concrete_Location=="Alaska" & pca10$Type=="Freshwater") | (pca10$Concrete_Location=="Alaska" | pca10$Concrete_Location=="East_Russia") & pca10$Type=="Marine"),]
test_pca10 = anti_join(pca10, train_pca10)
train_pca11 = pca11[which((pca11$Concrete_Location=="Alaska" & pca11$Type=="Freshwater") | (pca11$Concrete_Location=="Alaska" | pca11$Concrete_Location=="East_Russia") & pca11$Type=="Marine"),]
test_pca11 = anti_join(pca11, train_pca11)
train_pca12 = pca12[which((pca12$Concrete_Location=="Alaska" & pca12$Type=="Freshwater") | (pca12$Concrete_Location=="Alaska" | pca12$Concrete_Location=="East_Russia") & pca12$Type=="Marine"),]
test_pca12 = anti_join(pca12, train_pca12)
train_pca13 = pca13[which((pca13$Concrete_Location=="Alaska" & pca13$Type=="Freshwater") | (pca13$Concrete_Location=="Alaska" | pca13$Concrete_Location=="East_Russia") & pca13$Type=="Marine"),]
test_pca13 = anti_join(pca13, train_pca13)
train_pca16 = pca16[which((pca16$Concrete_Location=="Alaska" & pca16$Type=="Freshwater") | (pca16$Concrete_Location=="Alaska" | pca16$Concrete_Location=="East_Russia") & pca16$Type=="Marine"),]
test_pca16 = anti_join(pca16, train_pca16)
train_pca18 = pca18[which((pca18$Concrete_Location=="Alaska" & pca18$Type=="Freshwater") | (pca18$Concrete_Location=="Alaska" | pca18$Concrete_Location=="East_Russia") & pca18$Type=="Marine"),]
test_pca18 = anti_join(pca18, train_pca18)
train_pca20 = pca20[which((pca20$Concrete_Location=="Alaska" & pca20$Type=="Freshwater") | (pca20$Concrete_Location=="Alaska" | pca20$Concrete_Location=="East_Russia") & pca20$Type=="Marine"),]
test_pca20 = anti_join(pca20, train_pca20)
train_pca22 = pca22[which((pca22$Concrete_Location=="Alaska" & pca22$Type=="Freshwater") | (pca22$Concrete_Location=="Alaska" | pca22$Concrete_Location=="East_Russia") & pca22$Type=="Marine"),]
test_pca22 = anti_join(pca22, train_pca22)
train_pca25 = pca25[which((pca25$Concrete_Location=="Alaska" & pca25$Type=="Freshwater") | (pca25$Concrete_Location=="Alaska" | pca25$Concrete_Location=="East_Russia") & pca25$Type=="Marine"),]
test_pca25 = anti_join(pca25, train_pca25)
train_pca27 = pca27[which((pca27$Concrete_Location=="Alaska" & pca27$Type=="Freshwater") | (pca27$Concrete_Location=="Alaska" | pca27$Concrete_Location=="East_Russia") & pca27$Type=="Marine"),]
test_pca27 = anti_join(pca27, train_pca27)

# Preprocessing parameters
param5 = train_pca5 %>% preProcess(method=c("center","scale"))
param6 = train_pca6 %>% preProcess(method=c("center","scale"))
param10 = train_pca10 %>% preProcess(method=c("center","scale"))
param11 = train_pca11 %>% preProcess(method=c("center","scale"))
param12 = train_pca12 %>% preProcess(method=c("center","scale"))
param13 = train_pca13 %>% preProcess(method=c("center","scale"))
param16 = train_pca16 %>% preProcess(method=c("center","scale"))
param18 = train_pca18 %>% preProcess(method=c("center","scale"))
param20 = train_pca20 %>% preProcess(method=c("center","scale"))
param22 = train_pca22 %>% preProcess(method=c("center","scale"))
param25 = train_pca25 %>% preProcess(method=c("center","scale"))
param27 = train_pca27 %>% preProcess(method=c("center","scale"))

#### Model PC1 ####

# Make the model
model5 = qda(Type~V1, data=train_pca5)
model6 = qda(Type~V1, data=train_pca6)
model10 = qda(Type~V1, data=train_pca10)
model11 = qda(Type~V1, data=train_pca11)
model12 = qda(Type~V1, data=train_pca12)
model13 = qda(Type~V1, data=train_pca13)
model16 = qda(Type~V1, data=train_pca16)
model18 = qda(Type~V1, data=train_pca18)
model20 = qda(Type~V1, data=train_pca20)
model22 = qda(Type~V1, data=train_pca22)
model25 = qda(Type~V1, data=train_pca25)
model27 = qda(Type~V1, data=train_pca27)

# Make predictions
predictions5 = model5 %>% predict(test_pca5)
predictions6 = model6 %>% predict(test_pca6)
predictions10 = model10 %>% predict(test_pca10)
predictions11 = model11 %>% predict(test_pca11)
predictions12 = model12 %>% predict(test_pca12)
predictions13 = model13 %>% predict(test_pca13)
predictions16 = model16 %>% predict(test_pca16)
predictions18 = model18 %>% predict(test_pca18)
predictions20 = model20 %>% predict(test_pca20)
predictions22 = model22 %>% predict(test_pca22)
predictions25 = model25 %>% predict(test_pca25)
predictions27 = model27 %>% predict(test_pca27)

# Model accuracy
accuracy5 = mean(predictions5$class == test_pca5$Type)
accuracy6 = mean(predictions6$class == test_pca6$Type)
accuracy10 = mean(predictions10$class == test_pca10$Type)
accuracy11 = mean(predictions11$class == test_pca11$Type)
accuracy12 = mean(predictions12$class == test_pca12$Type)
accuracy13 = mean(predictions13$class == test_pca13$Type)
accuracy16 = mean(predictions16$class == test_pca16$Type)
accuracy18 = mean(predictions18$class == test_pca18$Type)
accuracy20 = mean(predictions20$class == test_pca20$Type)
accuracy22 = mean(predictions22$class == test_pca22$Type)
accuracy25 = mean(predictions25$class == test_pca25$Type)
accuracy27 = mean(predictions27$class == test_pca27$Type)

# Data to plot
data5 = as.data.frame(predictions5)
data5 = cbind(test_pca5$Individual, test_pca5$Population, test_pca5$Concrete_Location, test_pca5$Type, data5)
names(data5) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data6 = as.data.frame(predictions6)
data6 = cbind(test_pca6$Individual, test_pca6$Population, test_pca6$Concrete_Location, test_pca6$Type, data6)
names(data6) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data10 = as.data.frame(predictions10)
data10 = cbind(test_pca10$Individual, test_pca10$Population, test_pca10$Concrete_Location, test_pca10$Type, data10)
names(data10) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data11 = as.data.frame(predictions11)
data11 = cbind(test_pca11$Individual, test_pca11$Population, test_pca11$Concrete_Location, test_pca11$Type, data11)
names(data11) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data12 = as.data.frame(predictions12)
data12 = cbind(test_pca12$Individual, test_pca12$Population, test_pca12$Concrete_Location, test_pca12$Type, data12)
names(data12) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data13 = as.data.frame(predictions13)
data13 = cbind(test_pca13$Individual, test_pca13$Population, test_pca13$Concrete_Location, test_pca13$Type, data13)
names(data13) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data16 = as.data.frame(predictions16)
data16 = cbind(test_pca16$Individual, test_pca16$Population, test_pca16$Concrete_Location, test_pca16$Type, data16)
names(data16) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data18 = as.data.frame(predictions18)
data18 = cbind(test_pca18$Individual, test_pca18$Population, test_pca18$Concrete_Location, test_pca18$Type, data18)
names(data18) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data20 = as.data.frame(predictions20)
data20 = cbind(test_pca20$Individual, test_pca20$Population, test_pca20$Concrete_Location, test_pca20$Type, data20)
names(data20) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data22 = as.data.frame(predictions22)
data22 = cbind(test_pca22$Individual, test_pca22$Population, test_pca22$Concrete_Location, test_pca22$Type, data22)
names(data22) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data25 = as.data.frame(predictions25)
data25 = cbind(test_pca25$Individual, test_pca25$Population, test_pca25$Concrete_Location, test_pca25$Type, data25)
names(data25) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data27 = as.data.frame(predictions27)
data27 = cbind(test_pca27$Individual, test_pca27$Population, test_pca27$Concrete_Location, test_pca27$Type, data27)
names(data27) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")

# Order the population factors
data5$Population = factor(data5$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data6$Population = factor(data6$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data10$Population = factor(data10$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data11$Population = factor(data11$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data12$Population = factor(data12$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data13$Population = factor(data13$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data16$Population = factor(data16$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data18$Population = factor(data18$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data20$Population = factor(data20$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data22$Population = factor(data22$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data25$Population = factor(data25$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data27$Population = factor(data27$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 

# Data for heatmap
heatmap5 = data5
names(heatmap5)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap6 = data6
names(heatmap6)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap10 = data10
names(heatmap10)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap11 = data11
names(heatmap11)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap12 = data12
names(heatmap12)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap13 = data13
names(heatmap13)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap16 = data16
names(heatmap16)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap18 = data18
names(heatmap18)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap20 = data20
names(heatmap20)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap22 = data22
names(heatmap22)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap25 = data25
names(heatmap25)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")
heatmap27 = data27
names(heatmap27)[5:7] = c("Test_Type_1", "Posterior_Freshwater_1", "Posterior_Marine_1")

# Save the clusters that have the highest accuracy with 1 PC
finaldata11 = data11
finaldata13 = data13
finaldata18 = data18
finaldata27 = data27
finalaccuracy11 = accuracy11
finalaccuracy13 = accuracy13
finalaccuracy18 = accuracy18
finalaccuracy27 = accuracy27

# Plot points
plot5 = ggplot(data=data5, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.5) + 
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=data6, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=data10, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=data11, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=data12, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=data13, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=data16, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=data18, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=data20, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=data22, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=data25, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=data27, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1 Points.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

# Plot numbers
num5 = aggregate(Individual ~ Population + Test_Type, data = data5, FUN = length)
num5 = merge(data5[,c("Population", "Concrete_Location", "True_Type")], num5, by="Population", all=FALSE)
num5 = unique(num5)
num5$Test_Type = factor(num5$Test_Type, levels=c("Marine", "Freshwater"))
num6 = aggregate(Individual ~ Population + Test_Type, data = data6, FUN = length)
num6 = merge(data6[,c("Population", "Concrete_Location", "True_Type")], num6, by="Population", all=FALSE)
num6 = unique(num6)
num6$Test_Type = factor(num6$Test_Type, levels=c("Marine", "Freshwater"))
num10 = aggregate(Individual ~ Population + Test_Type, data = data10, FUN = length)
num10 = merge(data10[,c("Population", "Concrete_Location", "True_Type")], num10, by="Population", all=FALSE)
num10 = unique(num10)
num10$Test_Type = factor(num10$Test_Type, levels=c("Marine", "Freshwater"))
num11 = aggregate(Individual ~ Population + Test_Type, data = data11, FUN = length)
num11 = merge(data11[,c("Population", "Concrete_Location", "True_Type")], num11, by="Population", all=FALSE)
num11 = unique(num11)
num11$Test_Type = factor(num11$Test_Type, levels=c("Marine", "Freshwater"))
num12 = aggregate(Individual ~ Population + Test_Type, data = data12, FUN = length)
num12 = merge(data12[,c("Population", "Concrete_Location", "True_Type")], num12, by="Population", all=FALSE)
num12 = unique(num12)
num12$Test_Type = factor(num12$Test_Type, levels=c("Marine", "Freshwater"))
num13 = aggregate(Individual ~ Population + Test_Type, data = data13, FUN = length)
num13 = merge(data13[,c("Population", "Concrete_Location", "True_Type")], num13, by="Population", all=FALSE)
num13 = unique(num13)
num13$Test_Type = factor(num13$Test_Type, levels=c("Marine", "Freshwater"))
num16 = aggregate(Individual ~ Population + Test_Type, data = data16, FUN = length)
num16 = merge(data16[,c("Population", "Concrete_Location", "True_Type")], num16, by="Population", all=FALSE)
num16 = unique(num16)
num16$Test_Type = factor(num16$Test_Type, levels=c("Marine", "Freshwater"))
num18 = aggregate(Individual ~ Population + Test_Type, data = data18, FUN = length)
num18 = merge(data18[,c("Population", "Concrete_Location", "True_Type")], num18, by="Population", all=FALSE)
num18 = unique(num18)
num18$Test_Type = factor(num18$Test_Type, levels=c("Marine", "Freshwater"))
num20 = aggregate(Individual ~ Population + Test_Type, data = data20, FUN = length)
num20 = merge(data20[,c("Population", "Concrete_Location", "True_Type")], num20, by="Population", all=FALSE)
num20 = unique(num20)
num20$Test_Type = factor(num20$Test_Type, levels=c("Marine", "Freshwater"))
num22 = aggregate(Individual ~ Population + Test_Type, data = data22, FUN = length)
num22 = merge(data22[,c("Population", "Concrete_Location", "True_Type")], num22, by="Population", all=FALSE)
num22 = unique(num22)
num22$Test_Type = factor(num22$Test_Type, levels=c("Marine", "Freshwater"))
num25 = aggregate(Individual ~ Population + Test_Type, data = data25, FUN = length)
num25 = merge(data25[,c("Population", "Concrete_Location", "True_Type")], num25, by="Population", all=FALSE)
num25 = unique(num25)
num25$Test_Type = factor(num25$Test_Type, levels=c("Marine", "Freshwater"))
num27 = aggregate(Individual ~ Population + Test_Type, data = data27, FUN = length)
num27 = merge(data27[,c("Population", "Concrete_Location", "True_Type")], num27, by="Population", all=FALSE)
num27 = unique(num27)
num27$Test_Type = factor(num27$Test_Type, levels=c("Marine", "Freshwater"))

plot5 = ggplot(data=num5, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() +
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(color=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1, shape=16, linetype=0))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=num6, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=num10, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=num11, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=num12, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=num13, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=num16, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=num18, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=num20, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=num22, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=num25, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=num27, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1 Numbers.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

#### Model PC1+PC2 ####

# Make the model
model5 = qda(Type~V1+V2, data=train_pca5)
model6 = qda(Type~V1+V2, data=train_pca6)
model10 = qda(Type~V1+V2, data=train_pca10)
model11 = qda(Type~V1+V2, data=train_pca11)
model12 = qda(Type~V1+V2, data=train_pca12)
model13 = qda(Type~V1+V2, data=train_pca13)
model16 = qda(Type~V1+V2, data=train_pca16)
model18 = qda(Type~V1+V2, data=train_pca18)
model20 = qda(Type~V1+V2, data=train_pca20)
model22 = qda(Type~V1+V2, data=train_pca22)
model25 = qda(Type~V1+V2, data=train_pca25)
model27 = qda(Type~V1+V2, data=train_pca27)

# Make predictions
predictions5 = model5 %>% predict(test_pca5)
predictions6 = model6 %>% predict(test_pca6)
predictions10 = model10 %>% predict(test_pca10)
predictions11 = model11 %>% predict(test_pca11)
predictions12 = model12 %>% predict(test_pca12)
predictions13 = model13 %>% predict(test_pca13)
predictions16 = model16 %>% predict(test_pca16)
predictions18 = model18 %>% predict(test_pca18)
predictions20 = model20 %>% predict(test_pca20)
predictions22 = model22 %>% predict(test_pca22)
predictions25 = model25 %>% predict(test_pca25)
predictions27 = model27 %>% predict(test_pca27)

# Model accuracy
accuracy5 = mean(predictions5$class == test_pca5$Type)
accuracy6 = mean(predictions6$class == test_pca6$Type)
accuracy10 = mean(predictions10$class == test_pca10$Type)
accuracy11 = mean(predictions11$class == test_pca11$Type)
accuracy12 = mean(predictions12$class == test_pca12$Type)
accuracy13 = mean(predictions13$class == test_pca13$Type)
accuracy16 = mean(predictions16$class == test_pca16$Type)
accuracy18 = mean(predictions18$class == test_pca18$Type)
accuracy20 = mean(predictions20$class == test_pca20$Type)
accuracy22 = mean(predictions22$class == test_pca22$Type)
accuracy25 = mean(predictions25$class == test_pca25$Type)
accuracy27 = mean(predictions27$class == test_pca27$Type)

# Data to plot
data5 = as.data.frame(predictions5)
data5 = cbind(test_pca5$Individual, test_pca5$Population, test_pca5$Concrete_Location, test_pca5$Type, data5)
names(data5) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data6 = as.data.frame(predictions6)
data6 = cbind(test_pca6$Individual, test_pca6$Population, test_pca6$Concrete_Location, test_pca6$Type, data6)
names(data6) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data10 = as.data.frame(predictions10)
data10 = cbind(test_pca10$Individual, test_pca10$Population, test_pca10$Concrete_Location, test_pca10$Type, data10)
names(data10) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data11 = as.data.frame(predictions11)
data11 = cbind(test_pca11$Individual, test_pca11$Population, test_pca11$Concrete_Location, test_pca11$Type, data11)
names(data11) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data12 = as.data.frame(predictions12)
data12 = cbind(test_pca12$Individual, test_pca12$Population, test_pca12$Concrete_Location, test_pca12$Type, data12)
names(data12) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data13 = as.data.frame(predictions13)
data13 = cbind(test_pca13$Individual, test_pca13$Population, test_pca13$Concrete_Location, test_pca13$Type, data13)
names(data13) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data16 = as.data.frame(predictions16)
data16 = cbind(test_pca16$Individual, test_pca16$Population, test_pca16$Concrete_Location, test_pca16$Type, data16)
names(data16) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data18 = as.data.frame(predictions18)
data18 = cbind(test_pca18$Individual, test_pca18$Population, test_pca18$Concrete_Location, test_pca18$Type, data18)
names(data18) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data20 = as.data.frame(predictions20)
data20 = cbind(test_pca20$Individual, test_pca20$Population, test_pca20$Concrete_Location, test_pca20$Type, data20)
names(data20) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data22 = as.data.frame(predictions22)
data22 = cbind(test_pca22$Individual, test_pca22$Population, test_pca22$Concrete_Location, test_pca22$Type, data22)
names(data22) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data25 = as.data.frame(predictions25)
data25 = cbind(test_pca25$Individual, test_pca25$Population, test_pca25$Concrete_Location, test_pca25$Type, data25)
names(data25) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data27 = as.data.frame(predictions27)
data27 = cbind(test_pca27$Individual, test_pca27$Population, test_pca27$Concrete_Location, test_pca27$Type, data27)
names(data27) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")

# Order the population factors
data5$Population = factor(data5$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data6$Population = factor(data6$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data10$Population = factor(data10$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data11$Population = factor(data11$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data12$Population = factor(data12$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data13$Population = factor(data13$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data16$Population = factor(data16$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data18$Population = factor(data18$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data20$Population = factor(data20$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data22$Population = factor(data22$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data25$Population = factor(data25$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data27$Population = factor(data27$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 

# Data for heatmap
heatmap5$Test_Type_12 = data5$Test_Type
heatmap5$Posterior_Freshwater_12 = data5$Posterior_Freshwater
heatmap5$Posterior_Marine_12 = data5$Posterior_Marine
heatmap6$Test_Type_12 = data6$Test_Type
heatmap6$Posterior_Freshwater_12 = data6$Posterior_Freshwater
heatmap6$Posterior_Marine_12 = data6$Posterior_Marine
heatmap10$Test_Type_12 = data10$Test_Type
heatmap10$Posterior_Freshwater_12 = data10$Posterior_Freshwater
heatmap10$Posterior_Marine_12 = data10$Posterior_Marine
heatmap11$Test_Type_12 = data11$Test_Type
heatmap11$Posterior_Freshwater_12 = data11$Posterior_Freshwater
heatmap11$Posterior_Marine_12 = data11$Posterior_Marine
heatmap12$Test_Type_12 = data12$Test_Type
heatmap12$Posterior_Freshwater_12 = data12$Posterior_Freshwater
heatmap12$Posterior_Marine_12 = data12$Posterior_Marine
heatmap13$Test_Type_12 = data13$Test_Type
heatmap13$Posterior_Freshwater_12 = data13$Posterior_Freshwater
heatmap13$Posterior_Marine_12 = data13$Posterior_Marine
heatmap16$Test_Type_12 = data16$Test_Type
heatmap16$Posterior_Freshwater_12 = data16$Posterior_Freshwater
heatmap16$Posterior_Marine_12 = data16$Posterior_Marine
heatmap18$Test_Type_12 = data18$Test_Type
heatmap18$Posterior_Freshwater_12 = data18$Posterior_Freshwater
heatmap18$Posterior_Marine_12 = data18$Posterior_Marine
heatmap20$Test_Type_12 = data20$Test_Type
heatmap20$Posterior_Freshwater_12 = data20$Posterior_Freshwater
heatmap20$Posterior_Marine_12 = data20$Posterior_Marine
heatmap22$Test_Type_12 = data22$Test_Type
heatmap22$Posterior_Freshwater_12 = data22$Posterior_Freshwater
heatmap22$Posterior_Marine_12 = data22$Posterior_Marine
heatmap25$Test_Type_12 = data25$Test_Type
heatmap25$Posterior_Freshwater_12 = data25$Posterior_Freshwater
heatmap25$Posterior_Marine_12 = data25$Posterior_Marine
heatmap27$Test_Type_12 = data27$Test_Type
heatmap27$Posterior_Freshwater_12 = data27$Posterior_Freshwater
heatmap27$Posterior_Marine_12 = data27$Posterior_Marine

# Save the clusters that have the highest accuracy with 2 PC
finaldata6 = data6
finaldata12 = data12
finaldata20 = data20
finalaccuracy6 = accuracy6
finalaccuracy12 = accuracy12
finalaccuracy20 = accuracy20

# Plot points
plot5 = ggplot(data=data5, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.5) + 
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=data6, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=data10, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=data11, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=data12, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=data13, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=data16, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=data18, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=data20, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=data22, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=data25, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=data27, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1+PC2 Points.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

# Plot numbers
num5 = aggregate(Individual ~ Population + Test_Type, data = data5, FUN = length)
num5 = merge(data5[,c("Population", "Concrete_Location", "True_Type")], num5, by="Population", all=FALSE)
num5 = unique(num5)
num5$Test_Type = factor(num5$Test_Type, levels=c("Marine", "Freshwater"))
num6 = aggregate(Individual ~ Population + Test_Type, data = data6, FUN = length)
num6 = merge(data6[,c("Population", "Concrete_Location", "True_Type")], num6, by="Population", all=FALSE)
num6 = unique(num6)
num6$Test_Type = factor(num6$Test_Type, levels=c("Marine", "Freshwater"))
num10 = aggregate(Individual ~ Population + Test_Type, data = data10, FUN = length)
num10 = merge(data10[,c("Population", "Concrete_Location", "True_Type")], num10, by="Population", all=FALSE)
num10 = unique(num10)
num10$Test_Type = factor(num10$Test_Type, levels=c("Marine", "Freshwater"))
num11 = aggregate(Individual ~ Population + Test_Type, data = data11, FUN = length)
num11 = merge(data11[,c("Population", "Concrete_Location", "True_Type")], num11, by="Population", all=FALSE)
num11 = unique(num11)
num11$Test_Type = factor(num11$Test_Type, levels=c("Marine", "Freshwater"))
num12 = aggregate(Individual ~ Population + Test_Type, data = data12, FUN = length)
num12 = merge(data12[,c("Population", "Concrete_Location", "True_Type")], num12, by="Population", all=FALSE)
num12 = unique(num12)
num12$Test_Type = factor(num12$Test_Type, levels=c("Marine", "Freshwater"))
num13 = aggregate(Individual ~ Population + Test_Type, data = data13, FUN = length)
num13 = merge(data13[,c("Population", "Concrete_Location", "True_Type")], num13, by="Population", all=FALSE)
num13 = unique(num13)
num13$Test_Type = factor(num13$Test_Type, levels=c("Marine", "Freshwater"))
num16 = aggregate(Individual ~ Population + Test_Type, data = data16, FUN = length)
num16 = merge(data16[,c("Population", "Concrete_Location", "True_Type")], num16, by="Population", all=FALSE)
num16 = unique(num16)
num16$Test_Type = factor(num16$Test_Type, levels=c("Marine", "Freshwater"))
num18 = aggregate(Individual ~ Population + Test_Type, data = data18, FUN = length)
num18 = merge(data18[,c("Population", "Concrete_Location", "True_Type")], num18, by="Population", all=FALSE)
num18 = unique(num18)
num18$Test_Type = factor(num18$Test_Type, levels=c("Marine", "Freshwater"))
num20 = aggregate(Individual ~ Population + Test_Type, data = data20, FUN = length)
num20 = merge(data20[,c("Population", "Concrete_Location", "True_Type")], num20, by="Population", all=FALSE)
num20 = unique(num20)
num20$Test_Type = factor(num20$Test_Type, levels=c("Marine", "Freshwater"))
num22 = aggregate(Individual ~ Population + Test_Type, data = data22, FUN = length)
num22 = merge(data22[,c("Population", "Concrete_Location", "True_Type")], num22, by="Population", all=FALSE)
num22 = unique(num22)
num22$Test_Type = factor(num22$Test_Type, levels=c("Marine", "Freshwater"))
num25 = aggregate(Individual ~ Population + Test_Type, data = data25, FUN = length)
num25 = merge(data25[,c("Population", "Concrete_Location", "True_Type")], num25, by="Population", all=FALSE)
num25 = unique(num25)
num25$Test_Type = factor(num25$Test_Type, levels=c("Marine", "Freshwater"))
num27 = aggregate(Individual ~ Population + Test_Type, data = data27, FUN = length)
num27 = merge(data27[,c("Population", "Concrete_Location", "True_Type")], num27, by="Population", all=FALSE)
num27 = unique(num27)
num27$Test_Type = factor(num27$Test_Type, levels=c("Marine", "Freshwater"))

plot5 = ggplot(data=num5, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() +
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(color=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1, shape=16, linetype=0))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=num6, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=num10, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=num11, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=num12, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=num13, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=num16, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=num18, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=num20, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=num22, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=num25, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=num27, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1+PC2 Numbers.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

#### Model PC1+PC2+PC3 ####

# Make the model
model5 = qda(Type~V1+V2+V3, data=train_pca5)
model6 = qda(Type~V1+V2+V3, data=train_pca6)
model10 = qda(Type~V1+V2+V3, data=train_pca10)
model11 = qda(Type~V1+V2+V3, data=train_pca11)
model12 = qda(Type~V1+V2+V3, data=train_pca12)
model13 = qda(Type~V1+V2+V3, data=train_pca13)
model16 = qda(Type~V1+V2+V3, data=train_pca16)
model18 = qda(Type~V1+V2+V3, data=train_pca18)
model20 = qda(Type~V1+V2+V3, data=train_pca20)
model22 = qda(Type~V1+V2+V3, data=train_pca22)
model25 = qda(Type~V1+V2+V3, data=train_pca25)
model27 = qda(Type~V1+V2+V3, data=train_pca27)

# Make predictions
predictions5 = model5 %>% predict(test_pca5)
predictions6 = model6 %>% predict(test_pca6)
predictions10 = model10 %>% predict(test_pca10)
predictions11 = model11 %>% predict(test_pca11)
predictions12 = model12 %>% predict(test_pca12)
predictions13 = model13 %>% predict(test_pca13)
predictions16 = model16 %>% predict(test_pca16)
predictions18 = model18 %>% predict(test_pca18)
predictions20 = model20 %>% predict(test_pca20)
predictions22 = model22 %>% predict(test_pca22)
predictions25 = model25 %>% predict(test_pca25)
predictions27 = model27 %>% predict(test_pca27)

# Model accuracy
accuracy5 = mean(predictions5$class == test_pca5$Type)
accuracy6 = mean(predictions6$class == test_pca6$Type)
accuracy10 = mean(predictions10$class == test_pca10$Type)
accuracy11 = mean(predictions11$class == test_pca11$Type)
accuracy12 = mean(predictions12$class == test_pca12$Type)
accuracy13 = mean(predictions13$class == test_pca13$Type)
accuracy16 = mean(predictions16$class == test_pca16$Type)
accuracy18 = mean(predictions18$class == test_pca18$Type)
accuracy20 = mean(predictions20$class == test_pca20$Type)
accuracy22 = mean(predictions22$class == test_pca22$Type)
accuracy25 = mean(predictions25$class == test_pca25$Type)
accuracy27 = mean(predictions27$class == test_pca27$Type)

# Data to plot
data5 = as.data.frame(predictions5)
data5 = cbind(test_pca5$Individual, test_pca5$Population, test_pca5$Concrete_Location, test_pca5$Type, data5)
names(data5) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data6 = as.data.frame(predictions6)
data6 = cbind(test_pca6$Individual, test_pca6$Population, test_pca6$Concrete_Location, test_pca6$Type, data6)
names(data6) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data10 = as.data.frame(predictions10)
data10 = cbind(test_pca10$Individual, test_pca10$Population, test_pca10$Concrete_Location, test_pca10$Type, data10)
names(data10) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data11 = as.data.frame(predictions11)
data11 = cbind(test_pca11$Individual, test_pca11$Population, test_pca11$Concrete_Location, test_pca11$Type, data11)
names(data11) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data12 = as.data.frame(predictions12)
data12 = cbind(test_pca12$Individual, test_pca12$Population, test_pca12$Concrete_Location, test_pca12$Type, data12)
names(data12) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data13 = as.data.frame(predictions13)
data13 = cbind(test_pca13$Individual, test_pca13$Population, test_pca13$Concrete_Location, test_pca13$Type, data13)
names(data13) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data16 = as.data.frame(predictions16)
data16 = cbind(test_pca16$Individual, test_pca16$Population, test_pca16$Concrete_Location, test_pca16$Type, data16)
names(data16) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data18 = as.data.frame(predictions18)
data18 = cbind(test_pca18$Individual, test_pca18$Population, test_pca18$Concrete_Location, test_pca18$Type, data18)
names(data18) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data20 = as.data.frame(predictions20)
data20 = cbind(test_pca20$Individual, test_pca20$Population, test_pca20$Concrete_Location, test_pca20$Type, data20)
names(data20) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data22 = as.data.frame(predictions22)
data22 = cbind(test_pca22$Individual, test_pca22$Population, test_pca22$Concrete_Location, test_pca22$Type, data22)
names(data22) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data25 = as.data.frame(predictions25)
data25 = cbind(test_pca25$Individual, test_pca25$Population, test_pca25$Concrete_Location, test_pca25$Type, data25)
names(data25) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data27 = as.data.frame(predictions27)
data27 = cbind(test_pca27$Individual, test_pca27$Population, test_pca27$Concrete_Location, test_pca27$Type, data27)
names(data27) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")

# Order the population factors
data5$Population = factor(data5$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data6$Population = factor(data6$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data10$Population = factor(data10$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data11$Population = factor(data11$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data12$Population = factor(data12$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data13$Population = factor(data13$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data16$Population = factor(data16$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data18$Population = factor(data18$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data20$Population = factor(data20$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data22$Population = factor(data22$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data25$Population = factor(data25$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data27$Population = factor(data27$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 

# Data for heatmap
heatmap5$Test_Type_123 = data5$Test_Type
heatmap5$Posterior_Freshwater_123 = data5$Posterior_Freshwater
heatmap5$Posterior_Marine_123 = data5$Posterior_Marine
heatmap6$Test_Type_123 = data6$Test_Type
heatmap6$Posterior_Freshwater_123 = data6$Posterior_Freshwater
heatmap6$Posterior_Marine_123 = data6$Posterior_Marine
heatmap10$Test_Type_123 = data10$Test_Type
heatmap10$Posterior_Freshwater_123 = data10$Posterior_Freshwater
heatmap10$Posterior_Marine_123 = data10$Posterior_Marine
heatmap11$Test_Type_123 = data11$Test_Type
heatmap11$Posterior_Freshwater_123 = data11$Posterior_Freshwater
heatmap11$Posterior_Marine_123 = data11$Posterior_Marine
heatmap12$Test_Type_123 = data12$Test_Type
heatmap12$Posterior_Freshwater_123 = data12$Posterior_Freshwater
heatmap12$Posterior_Marine_123 = data12$Posterior_Marine
heatmap13$Test_Type_123 = data13$Test_Type
heatmap13$Posterior_Freshwater_123 = data13$Posterior_Freshwater
heatmap13$Posterior_Marine_123 = data13$Posterior_Marine
heatmap16$Test_Type_123 = data16$Test_Type
heatmap16$Posterior_Freshwater_123 = data16$Posterior_Freshwater
heatmap16$Posterior_Marine_123 = data16$Posterior_Marine
heatmap18$Test_Type_123 = data18$Test_Type
heatmap18$Posterior_Freshwater_123 = data18$Posterior_Freshwater
heatmap18$Posterior_Marine_123 = data18$Posterior_Marine
heatmap20$Test_Type_123 = data20$Test_Type
heatmap20$Posterior_Freshwater_123 = data20$Posterior_Freshwater
heatmap20$Posterior_Marine_123 = data20$Posterior_Marine
heatmap22$Test_Type_123 = data22$Test_Type
heatmap22$Posterior_Freshwater_123 = data22$Posterior_Freshwater
heatmap22$Posterior_Marine_123 = data22$Posterior_Marine
heatmap25$Test_Type_13 = data25$Test_Type
heatmap25$Posterior_Freshwater_123 = data25$Posterior_Freshwater
heatmap25$Posterior_Marine_123 = data25$Posterior_Marine
heatmap27$Test_Type_123 = data27$Test_Type
heatmap27$Posterior_Freshwater_123 = data27$Posterior_Freshwater
heatmap27$Posterior_Marine_123 = data27$Posterior_Marine

# Save the clusters that have the highest accuracy with 3 PC
finaldata16 = data16
finaldata22 = data22
finaldata25 = data25
finalaccuracy16 = accuracy16
finalaccuracy22 = accuracy22
finalaccuracy25 = accuracy25

# Plot points
plot5 = ggplot(data=data5, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.5) + 
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=data6, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=data10, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=data11, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=data12, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=data13, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=data16, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=data18, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=data20, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=data22, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=data25, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=data27, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1+PC2+PC3 Points.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

# Plot numbers
num5 = aggregate(Individual ~ Population + Test_Type, data = data5, FUN = length)
num5 = merge(data5[,c("Population", "Concrete_Location", "True_Type")], num5, by="Population", all=FALSE)
num5 = unique(num5)
num5$Test_Type = factor(num5$Test_Type, levels=c("Marine", "Freshwater"))
num6 = aggregate(Individual ~ Population + Test_Type, data = data6, FUN = length)
num6 = merge(data6[,c("Population", "Concrete_Location", "True_Type")], num6, by="Population", all=FALSE)
num6 = unique(num6)
num6$Test_Type = factor(num6$Test_Type, levels=c("Marine", "Freshwater"))
num10 = aggregate(Individual ~ Population + Test_Type, data = data10, FUN = length)
num10 = merge(data10[,c("Population", "Concrete_Location", "True_Type")], num10, by="Population", all=FALSE)
num10 = unique(num10)
num10$Test_Type = factor(num10$Test_Type, levels=c("Marine", "Freshwater"))
num11 = aggregate(Individual ~ Population + Test_Type, data = data11, FUN = length)
num11 = merge(data11[,c("Population", "Concrete_Location", "True_Type")], num11, by="Population", all=FALSE)
num11 = unique(num11)
num11$Test_Type = factor(num11$Test_Type, levels=c("Marine", "Freshwater"))
num12 = aggregate(Individual ~ Population + Test_Type, data = data12, FUN = length)
num12 = merge(data12[,c("Population", "Concrete_Location", "True_Type")], num12, by="Population", all=FALSE)
num12 = unique(num12)
num12$Test_Type = factor(num12$Test_Type, levels=c("Marine", "Freshwater"))
num13 = aggregate(Individual ~ Population + Test_Type, data = data13, FUN = length)
num13 = merge(data13[,c("Population", "Concrete_Location", "True_Type")], num13, by="Population", all=FALSE)
num13 = unique(num13)
num13$Test_Type = factor(num13$Test_Type, levels=c("Marine", "Freshwater"))
num16 = aggregate(Individual ~ Population + Test_Type, data = data16, FUN = length)
num16 = merge(data16[,c("Population", "Concrete_Location", "True_Type")], num16, by="Population", all=FALSE)
num16 = unique(num16)
num16$Test_Type = factor(num16$Test_Type, levels=c("Marine", "Freshwater"))
num18 = aggregate(Individual ~ Population + Test_Type, data = data18, FUN = length)
num18 = merge(data18[,c("Population", "Concrete_Location", "True_Type")], num18, by="Population", all=FALSE)
num18 = unique(num18)
num18$Test_Type = factor(num18$Test_Type, levels=c("Marine", "Freshwater"))
num20 = aggregate(Individual ~ Population + Test_Type, data = data20, FUN = length)
num20 = merge(data20[,c("Population", "Concrete_Location", "True_Type")], num20, by="Population", all=FALSE)
num20 = unique(num20)
num20$Test_Type = factor(num20$Test_Type, levels=c("Marine", "Freshwater"))
num22 = aggregate(Individual ~ Population + Test_Type, data = data22, FUN = length)
num22 = merge(data22[,c("Population", "Concrete_Location", "True_Type")], num22, by="Population", all=FALSE)
num22 = unique(num22)
num22$Test_Type = factor(num22$Test_Type, levels=c("Marine", "Freshwater"))
num25 = aggregate(Individual ~ Population + Test_Type, data = data25, FUN = length)
num25 = merge(data25[,c("Population", "Concrete_Location", "True_Type")], num25, by="Population", all=FALSE)
num25 = unique(num25)
num25$Test_Type = factor(num25$Test_Type, levels=c("Marine", "Freshwater"))
num27 = aggregate(Individual ~ Population + Test_Type, data = data27, FUN = length)
num27 = merge(data27[,c("Population", "Concrete_Location", "True_Type")], num27, by="Population", all=FALSE)
num27 = unique(num27)
num27$Test_Type = factor(num27$Test_Type, levels=c("Marine", "Freshwater"))

plot5 = ggplot(data=num5, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() +
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(color=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1, shape=16, linetype=0))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=num6, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=num10, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=num11, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=num12, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=num13, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=num16, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=num18, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=num20, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=num22, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=num25, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=num27, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1+PC2+PC3 Numbers.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

#### Model PC1+PC2+PC3+PC4 ####

# Make the model
model5 = qda(Type~V1+V2+V3+V4, data=train_pca5)
model6 = qda(Type~V1+V2+V3+V4, data=train_pca6)
model10 = qda(Type~V1+V2+V3+V4, data=train_pca10)
model11 = qda(Type~V1+V2+V3+V4, data=train_pca11)
model12 = qda(Type~V1+V2+V3+V4, data=train_pca12)
model13 = qda(Type~V1+V2+V3+V4, data=train_pca13)
model16 = qda(Type~V1+V2+V3+V4, data=train_pca16)
model18 = qda(Type~V1+V2+V3+V4, data=train_pca18)
model20 = qda(Type~V1+V2+V3+V4, data=train_pca20)
model22 = qda(Type~V1+V2+V3+V4, data=train_pca22)
model25 = qda(Type~V1+V2+V3+V4, data=train_pca25)
model27 = qda(Type~V1+V2+V3+V4, data=train_pca27)

# Make predictions
predictions5 = model5 %>% predict(test_pca5)
predictions6 = model6 %>% predict(test_pca6)
predictions10 = model10 %>% predict(test_pca10)
predictions11 = model11 %>% predict(test_pca11)
predictions12 = model12 %>% predict(test_pca12)
predictions13 = model13 %>% predict(test_pca13)
predictions16 = model16 %>% predict(test_pca16)
predictions18 = model18 %>% predict(test_pca18)
predictions20 = model20 %>% predict(test_pca20)
predictions22 = model22 %>% predict(test_pca22)
predictions25 = model25 %>% predict(test_pca25)
predictions27 = model27 %>% predict(test_pca27)

# Model accuracy
accuracy5 = mean(predictions5$class == test_pca5$Type)
accuracy6 = mean(predictions6$class == test_pca6$Type)
accuracy10 = mean(predictions10$class == test_pca10$Type)
accuracy11 = mean(predictions11$class == test_pca11$Type)
accuracy12 = mean(predictions12$class == test_pca12$Type)
accuracy13 = mean(predictions13$class == test_pca13$Type)
accuracy16 = mean(predictions16$class == test_pca16$Type)
accuracy18 = mean(predictions18$class == test_pca18$Type)
accuracy20 = mean(predictions20$class == test_pca20$Type)
accuracy22 = mean(predictions22$class == test_pca22$Type)
accuracy25 = mean(predictions25$class == test_pca25$Type)
accuracy27 = mean(predictions27$class == test_pca27$Type)

# Data to plot
data5 = as.data.frame(predictions5)
data5 = cbind(test_pca5$Individual, test_pca5$Population, test_pca5$Concrete_Location, test_pca5$Type, data5)
names(data5) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data6 = as.data.frame(predictions6)
data6 = cbind(test_pca6$Individual, test_pca6$Population, test_pca6$Concrete_Location, test_pca6$Type, data6)
names(data6) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data10 = as.data.frame(predictions10)
data10 = cbind(test_pca10$Individual, test_pca10$Population, test_pca10$Concrete_Location, test_pca10$Type, data10)
names(data10) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data11 = as.data.frame(predictions11)
data11 = cbind(test_pca11$Individual, test_pca11$Population, test_pca11$Concrete_Location, test_pca11$Type, data11)
names(data11) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data12 = as.data.frame(predictions12)
data12 = cbind(test_pca12$Individual, test_pca12$Population, test_pca12$Concrete_Location, test_pca12$Type, data12)
names(data12) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data13 = as.data.frame(predictions13)
data13 = cbind(test_pca13$Individual, test_pca13$Population, test_pca13$Concrete_Location, test_pca13$Type, data13)
names(data13) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data16 = as.data.frame(predictions16)
data16 = cbind(test_pca16$Individual, test_pca16$Population, test_pca16$Concrete_Location, test_pca16$Type, data16)
names(data16) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data18 = as.data.frame(predictions18)
data18 = cbind(test_pca18$Individual, test_pca18$Population, test_pca18$Concrete_Location, test_pca18$Type, data18)
names(data18) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data20 = as.data.frame(predictions20)
data20 = cbind(test_pca20$Individual, test_pca20$Population, test_pca20$Concrete_Location, test_pca20$Type, data20)
names(data20) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data22 = as.data.frame(predictions22)
data22 = cbind(test_pca22$Individual, test_pca22$Population, test_pca22$Concrete_Location, test_pca22$Type, data22)
names(data22) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data25 = as.data.frame(predictions25)
data25 = cbind(test_pca25$Individual, test_pca25$Population, test_pca25$Concrete_Location, test_pca25$Type, data25)
names(data25) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")
data27 = as.data.frame(predictions27)
data27 = cbind(test_pca27$Individual, test_pca27$Population, test_pca27$Concrete_Location, test_pca27$Type, data27)
names(data27) = c("Individual", "Population", "Concrete_Location", "True_Type", "Test_Type", "Posterior_Freshwater", "Posterior_Marine")

# Order the population factors
data5$Population = factor(data5$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data6$Population = factor(data6$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data10$Population = factor(data10$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data11$Population = factor(data11$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data12$Population = factor(data12$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data13$Population = factor(data13$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data16$Population = factor(data16$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data18$Population = factor(data18$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data20$Population = factor(data20$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data22$Population = factor(data22$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data25$Population = factor(data25$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 
data27$Population = factor(data27$Population, levels=c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "NOR-MYR-GA", "NOR-KVA-GA", "NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MUR", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 

# Data for heatmap
heatmap5$Test_Type_1234 = data5$Test_Type
heatmap5$Posterior_Freshwater_1234 = data5$Posterior_Freshwater
heatmap5$Posterior_Marine_1234 = data5$Posterior_Marine
heatmap6$Test_Type_1234 = data6$Test_Type
heatmap6$Posterior_Freshwater_1234 = data6$Posterior_Freshwater
heatmap6$Posterior_Marine_1234 = data6$Posterior_Marine
heatmap10$Test_Type_1234 = data10$Test_Type
heatmap10$Posterior_Freshwater_1234 = data10$Posterior_Freshwater
heatmap10$Posterior_Marine_1234 = data10$Posterior_Marine
heatmap11$Test_Type_1234 = data11$Test_Type
heatmap11$Posterior_Freshwater_1234 = data11$Posterior_Freshwater
heatmap11$Posterior_Marine_1234 = data11$Posterior_Marine
heatmap12$Test_Type_1234 = data12$Test_Type
heatmap12$Posterior_Freshwater_1234 = data12$Posterior_Freshwater
heatmap12$Posterior_Marine_1234 = data12$Posterior_Marine
heatmap13$Test_Type_1234 = data13$Test_Type
heatmap13$Posterior_Freshwater_1234 = data13$Posterior_Freshwater
heatmap13$Posterior_Marine_1234 = data13$Posterior_Marine
heatmap16$Test_Type_1234 = data16$Test_Type
heatmap16$Posterior_Freshwater_1234 = data16$Posterior_Freshwater
heatmap16$Posterior_Marine_1234 = data16$Posterior_Marine
heatmap18$Test_Type_1234 = data18$Test_Type
heatmap18$Posterior_Freshwater_1234 = data18$Posterior_Freshwater
heatmap18$Posterior_Marine_1234 = data18$Posterior_Marine
heatmap20$Test_Type_1234 = data20$Test_Type
heatmap20$Posterior_Freshwater_1234 = data20$Posterior_Freshwater
heatmap20$Posterior_Marine_1234 = data20$Posterior_Marine
heatmap22$Test_Type_1234 = data22$Test_Type
heatmap22$Posterior_Freshwater_1243 = data22$Posterior_Freshwater
heatmap22$Posterior_Marine_1234 = data22$Posterior_Marine
heatmap25$Test_Type_1234 = data25$Test_Type
heatmap25$Posterior_Freshwater_1234 = data25$Posterior_Freshwater
heatmap25$Posterior_Marine_1234 = data25$Posterior_Marine
heatmap27$Test_Type_1234 = data27$Test_Type
heatmap27$Posterior_Freshwater_1234 = data27$Posterior_Freshwater
heatmap27$Posterior_Marine_1234 = data27$Posterior_Marine

# Save the clusters that have the highest accuracy with 2 PC
finaldata5 = data5
finaldata10 = data10
finalaccuracy5 = accuracy5
finalaccuracy10 = accuracy10

# Plot points
plot5 = ggplot(data=data5, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.5) + 
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=data6, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=data10, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=data11, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=data12, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=data13, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=data16, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=data18, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=data20, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=data22, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=data25, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=data27, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type), alpha=0.8) + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1+PC2+PC3+PC4 Points.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

# Plot numbers
num5 = aggregate(Individual ~ Population + Test_Type, data = data5, FUN = length)
num5 = merge(data5[,c("Population", "Concrete_Location", "True_Type")], num5, by="Population", all=FALSE)
num5 = unique(num5)
num5$Test_Type = factor(num5$Test_Type, levels=c("Marine", "Freshwater"))
num6 = aggregate(Individual ~ Population + Test_Type, data = data6, FUN = length)
num6 = merge(data6[,c("Population", "Concrete_Location", "True_Type")], num6, by="Population", all=FALSE)
num6 = unique(num6)
num6$Test_Type = factor(num6$Test_Type, levels=c("Marine", "Freshwater"))
num10 = aggregate(Individual ~ Population + Test_Type, data = data10, FUN = length)
num10 = merge(data10[,c("Population", "Concrete_Location", "True_Type")], num10, by="Population", all=FALSE)
num10 = unique(num10)
num10$Test_Type = factor(num10$Test_Type, levels=c("Marine", "Freshwater"))
num11 = aggregate(Individual ~ Population + Test_Type, data = data11, FUN = length)
num11 = merge(data11[,c("Population", "Concrete_Location", "True_Type")], num11, by="Population", all=FALSE)
num11 = unique(num11)
num11$Test_Type = factor(num11$Test_Type, levels=c("Marine", "Freshwater"))
num12 = aggregate(Individual ~ Population + Test_Type, data = data12, FUN = length)
num12 = merge(data12[,c("Population", "Concrete_Location", "True_Type")], num12, by="Population", all=FALSE)
num12 = unique(num12)
num12$Test_Type = factor(num12$Test_Type, levels=c("Marine", "Freshwater"))
num13 = aggregate(Individual ~ Population + Test_Type, data = data13, FUN = length)
num13 = merge(data13[,c("Population", "Concrete_Location", "True_Type")], num13, by="Population", all=FALSE)
num13 = unique(num13)
num13$Test_Type = factor(num13$Test_Type, levels=c("Marine", "Freshwater"))
num16 = aggregate(Individual ~ Population + Test_Type, data = data16, FUN = length)
num16 = merge(data16[,c("Population", "Concrete_Location", "True_Type")], num16, by="Population", all=FALSE)
num16 = unique(num16)
num16$Test_Type = factor(num16$Test_Type, levels=c("Marine", "Freshwater"))
num18 = aggregate(Individual ~ Population + Test_Type, data = data18, FUN = length)
num18 = merge(data18[,c("Population", "Concrete_Location", "True_Type")], num18, by="Population", all=FALSE)
num18 = unique(num18)
num18$Test_Type = factor(num18$Test_Type, levels=c("Marine", "Freshwater"))
num20 = aggregate(Individual ~ Population + Test_Type, data = data20, FUN = length)
num20 = merge(data20[,c("Population", "Concrete_Location", "True_Type")], num20, by="Population", all=FALSE)
num20 = unique(num20)
num20$Test_Type = factor(num20$Test_Type, levels=c("Marine", "Freshwater"))
num22 = aggregate(Individual ~ Population + Test_Type, data = data22, FUN = length)
num22 = merge(data22[,c("Population", "Concrete_Location", "True_Type")], num22, by="Population", all=FALSE)
num22 = unique(num22)
num22$Test_Type = factor(num22$Test_Type, levels=c("Marine", "Freshwater"))
num25 = aggregate(Individual ~ Population + Test_Type, data = data25, FUN = length)
num25 = merge(data25[,c("Population", "Concrete_Location", "True_Type")], num25, by="Population", all=FALSE)
num25 = unique(num25)
num25$Test_Type = factor(num25$Test_Type, levels=c("Marine", "Freshwater"))
num27 = aggregate(Individual ~ Population + Test_Type, data = data27, FUN = length)
num27 = merge(data27[,c("Population", "Concrete_Location", "True_Type")], num27, by="Population", all=FALSE)
num27 = unique(num27)
num27$Test_Type = factor(num27$Test_Type, levels=c("Marine", "Freshwater"))

plot5 = ggplot(data=num5, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() +
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy5,2),")")) +
  scale_x_discrete(limits = rev(levels(data5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(color=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1, shape=16, linetype=0))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=num6, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy6,2),")")) +
  scale_x_discrete(limits = rev(levels(data6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=num10, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy10,2),")")) +
  scale_x_discrete(limits = rev(levels(data10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=num11, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy11,2),")")) +
  scale_x_discrete(limits = rev(levels(data11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=num12, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy12,2),")")) +
  scale_x_discrete(limits = rev(levels(data12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=num13, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy13,2),")")) +
  scale_x_discrete(limits = rev(levels(data13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=num16, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy16,2),")")) +
  scale_x_discrete(limits = rev(levels(data16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=num18, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy18,2),")")) +
  scale_x_discrete(limits = rev(levels(data18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=num20, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy20,2),")")) +
  scale_x_discrete(limits = rev(levels(data20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=num22, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy22,2),")")) +
  scale_x_discrete(limits = rev(levels(data22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=num25, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy25,2),")")) +
  scale_x_discrete(limits = rev(levels(data25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=num27, aes(x=Population, y=Test_Type, color=Concrete_Location, label=Individual)) + 
  geom_text() + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (accuracy: ", round(accuracy27,2),")")) +
  scale_x_discrete(limits = rev(levels(data27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Quadratic Discriminant Analysis PC1+PC2+PC3+PC4 Numbers.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

#### Final Plots ####

plot5 = ggplot(data=finaldata5, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (4PC - accuracy: ", round(finalaccuracy5*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata5$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot6 = ggplot(data=finaldata6, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (2PC - accuracy: ", round(finalaccuracy6*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata6$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot10 = ggplot(data=finaldata10, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (4PC - accuracy: ", round(finalaccuracy10*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata10$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot11 = ggplot(data=finaldata11, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (1PC - accuracy: ", round(finalaccuracy11*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata11$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot12 = ggplot(data=finaldata12, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (2PC - accuracy: ", round(finalaccuracy12*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata12$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot13 = ggplot(data=finaldata13, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (1PC - accuracy: ", round(finalaccuracy13*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata13$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot16 = ggplot(data=finaldata16, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (3PC - accuracy: ", round(finalaccuracy16*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata16$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot18 = ggplot(data=finaldata18, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (1PC - accuracy: ", round(finalaccuracy18*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata18$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot20 = ggplot(data=finaldata20, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (2PC - accuracy: ", round(finalaccuracy20*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata20$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot22 = ggplot(data=finaldata22, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (3PC - accuracy: ", round(finalaccuracy22*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata22$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot25 = ggplot(data=finaldata25, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (3PC - accuracy: ", round(finalaccuracy25*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata25$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

plot27 = ggplot(data=finaldata27, aes(x=Population, y=Posterior_Freshwater, color=Concrete_Location)) + 
  geom_point(aes(shape=True_Type, alpha=True_Type), size=4) + 
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Population") + 
  ylab(paste0("Probablity of Freshwater (1PC - accuracy: ", round(finalaccuracy27*100,2),"%)")) +
  scale_x_discrete(limits = rev(levels(finaldata27$Population))) +
  scale_color_manual(name="Geographic Region", values=c("Adriatic_Sea"="orange", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="chartreuse3", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4", "Mur_River"="cyan2"),
                     labels=c("Adriatic_Sea"="Adriatic Sea", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea", "Mur_River"="Mur River")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.3, "Marine"=0.7)) +
  coord_flip() +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="True Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.y=element_text(color=c("skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "skyblue1", "navy", "navy", "navy", "navy", "navy", "navy", "navy")))

png("Final Plot.png", units="in", width=15, height=20, res=900)
ggarrange(plot5,
          plot6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot11,
          plot12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot16,
          plot18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot22,
          plot25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          plot27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()

#### QDA Heatmap Proportion Ecotype per Population ####

# Proportion of each ecotype
totalindividuals5 = aggregate(Individual~Population, data=finaldata5, FUN=length)
names(totalindividuals5)[2] = "Total"
ecotypeindividuals5 = aggregate(Individual~Population+Test_Type, data=finaldata5, FUN=length)
ecotypeindividuals5 = dcast(ecotypeindividuals5, Population~Test_Type)
proportion5 = merge(totalindividuals5, ecotypeindividuals5, by="Population")
proportion5$prop_freshwater = proportion5$Freshwater/proportion5$Total
proportion5$prop_marine = proportion5$Marine/proportion5$Total
proportion5[is.na(proportion5)] = 0

totalindividuals6 = aggregate(Individual~Population, data=finaldata6, FUN=length)
names(totalindividuals6)[2] = "Total"
ecotypeindividuals6 = aggregate(Individual~Population+Test_Type, data=finaldata6, FUN=length)
ecotypeindividuals6 = dcast(ecotypeindividuals6, Population~Test_Type)
proportion6 = merge(totalindividuals6, ecotypeindividuals6, by="Population")
proportion6$prop_freshwater = proportion6$Freshwater/proportion6$Total
proportion6$prop_marine = proportion6$Marine/proportion6$Total
proportion6[is.na(proportion6)] = 0

totalindividuals10 = aggregate(Individual~Population, data=finaldata10, FUN=length)
names(totalindividuals10)[2] = "Total"
ecotypeindividuals10 = aggregate(Individual~Population+Test_Type, data=finaldata10, FUN=length)
ecotypeindividuals10 = dcast(ecotypeindividuals10, Population~Test_Type)
proportion10 = merge(totalindividuals10, ecotypeindividuals10, by="Population")
proportion10$prop_freshwater = proportion10$Freshwater/proportion10$Total
proportion10$prop_marine = proportion10$Marine/proportion10$Total
proportion10[is.na(proportion10)] = 0

totalindividuals11 = aggregate(Individual~Population, data=finaldata11, FUN=length)
names(totalindividuals11)[2] = "Total"
ecotypeindividuals11 = aggregate(Individual~Population+Test_Type, data=finaldata11, FUN=length)
ecotypeindividuals11 = dcast(ecotypeindividuals11, Population~Test_Type)
proportion11 = merge(totalindividuals11, ecotypeindividuals11, by="Population")
proportion11$prop_freshwater = proportion11$Freshwater/proportion11$Total
proportion11$prop_marine = proportion11$Marine/proportion11$Total
proportion11[is.na(proportion11)] = 0

totalindividuals12 = aggregate(Individual~Population, data=finaldata12, FUN=length)
names(totalindividuals12)[2] = "Total"
ecotypeindividuals12 = aggregate(Individual~Population+Test_Type, data=finaldata12, FUN=length)
ecotypeindividuals12 = dcast(ecotypeindividuals12, Population~Test_Type)
proportion12 = merge(totalindividuals12, ecotypeindividuals12, by="Population")
proportion12$prop_freshwater = proportion12$Freshwater/proportion12$Total
proportion12$prop_marine = proportion12$Marine/proportion12$Total
proportion12[is.na(proportion12)] = 0

totalindividuals13 = aggregate(Individual~Population, data=finaldata13, FUN=length)
names(totalindividuals13)[2] = "Total"
ecotypeindividuals13 = aggregate(Individual~Population+Test_Type, data=finaldata13, FUN=length)
ecotypeindividuals13 = dcast(ecotypeindividuals13, Population~Test_Type)
proportion13 = merge(totalindividuals13, ecotypeindividuals13, by="Population")
proportion13$prop_freshwater = proportion13$Freshwater/proportion13$Total
proportion13$prop_marine = proportion13$Marine/proportion13$Total
proportion13[is.na(proportion13)] = 0

totalindividuals16 = aggregate(Individual~Population, data=finaldata16, FUN=length)
names(totalindividuals16)[2] = "Total"
ecotypeindividuals16 = aggregate(Individual~Population+Test_Type, data=finaldata16, FUN=length)
ecotypeindividuals16 = dcast(ecotypeindividuals16, Population~Test_Type)
proportion16 = merge(totalindividuals16, ecotypeindividuals16, by="Population")
proportion16$prop_freshwater = proportion16$Freshwater/proportion16$Total
proportion16$prop_marine = proportion16$Marine/proportion16$Total
proportion16[is.na(proportion16)] = 0

totalindividuals18 = aggregate(Individual~Population, data=finaldata18, FUN=length)
names(totalindividuals18)[2] = "Total"
ecotypeindividuals18 = aggregate(Individual~Population+Test_Type, data=finaldata18, FUN=length)
ecotypeindividuals18 = dcast(ecotypeindividuals18, Population~Test_Type)
proportion18 = merge(totalindividuals18, ecotypeindividuals18, by="Population")
proportion18$prop_freshwater = proportion18$Freshwater/proportion18$Total
proportion18$prop_marine = proportion18$Marine/proportion18$Total
proportion18[is.na(proportion18)] = 0

totalindividuals20 = aggregate(Individual~Population, data=finaldata20, FUN=length)
names(totalindividuals20)[2] = "Total"
ecotypeindividuals20 = aggregate(Individual~Population+Test_Type, data=finaldata20, FUN=length)
ecotypeindividuals20 = dcast(ecotypeindividuals20, Population~Test_Type)
proportion20 = merge(totalindividuals20, ecotypeindividuals20, by="Population")
proportion20$prop_freshwater = proportion20$Freshwater/proportion20$Total
proportion20$prop_marine = proportion20$Marine/proportion20$Total
proportion20[is.na(proportion20)] = 0

totalindividuals22 = aggregate(Individual~Population, data=finaldata22, FUN=length)
names(totalindividuals22)[2] = "Total"
ecotypeindividuals22 = aggregate(Individual~Population+Test_Type, data=finaldata22, FUN=length)
ecotypeindividuals22 = dcast(ecotypeindividuals22, Population~Test_Type)
proportion22 = merge(totalindividuals22, ecotypeindividuals22, by="Population")
proportion22$prop_freshwater = proportion22$Freshwater/proportion22$Total
proportion22$prop_marine = proportion22$Marine/proportion22$Total
proportion22[is.na(proportion22)] = 0

totalindividuals25 = aggregate(Individual~Population, data=finaldata25, FUN=length)
names(totalindividuals25)[2] = "Total"
ecotypeindividuals25 = aggregate(Individual~Population+Test_Type, data=finaldata25, FUN=length)
ecotypeindividuals25 = dcast(ecotypeindividuals25, Population~Test_Type)
proportion25 = merge(totalindividuals25, ecotypeindividuals25, by="Population")
proportion25$prop_freshwater = proportion25$Freshwater/proportion25$Total
proportion25$prop_marine = proportion25$Marine/proportion25$Total
proportion25[is.na(proportion25)] = 0

totalindividuals27 = aggregate(Individual~Population, data=finaldata27, FUN=length)
names(totalindividuals27)[2] = "Total"
ecotypeindividuals27 = aggregate(Individual~Population+Test_Type, data=finaldata27, FUN=length)
ecotypeindividuals27 = dcast(ecotypeindividuals27, Population~Test_Type)
proportion27 = merge(totalindividuals27, ecotypeindividuals27, by="Population")
proportion27$prop_freshwater = proportion27$Freshwater/proportion27$Total
proportion27$prop_marine = proportion27$Marine/proportion27$Total
proportion27[is.na(proportion27)] = 0

# Create a variable to define the cluster and remove the prob of marine
proportion5$prop_marine = NULL
proportion5 = proportion5[,-c(2:4)]
names(proportion5)[2] = "Pfreshwater5"
proportion6$prop_marine = NULL
proportion6 = proportion6[,-c(2:4)]
names(proportion6)[2] = "Pfreshwater6"
proportion10$prop_marine = NULL
proportion10 = proportion10[,-c(2:4)]
names(proportion10)[2] = "Pfreshwater10"
proportion11$prop_marine = NULL
proportion11 = proportion11[,-c(2:4)]
names(proportion11)[2] = "Pfreshwater11"
proportion12$prop_marine = NULL
proportion12 = proportion12[,-c(2:4)]
names(proportion12)[2] = "Pfreshwater12"
proportion13$prop_marine = NULL
proportion13 = proportion13[,-c(2:4)]
names(proportion13)[2] = "Pfreshwater13"
proportion16$prop_marine = NULL
proportion16 = proportion16[,-c(2:4)]
names(proportion16)[2] = "Pfreshwater16"
proportion18$prop_marine = NULL
proportion18 = proportion18[,-c(2:4)]
names(proportion18)[2] = "Pfreshwater18"
proportion20$prop_marine = NULL
proportion20 = proportion20[,-c(2:4)]
names(proportion20)[2] = "Pfreshwater20"
proportion22$prop_marine = NULL
proportion22 = proportion22[,-c(2:4)]
names(proportion22)[2] = "Pfreshwater22"
proportion25$prop_marine = NULL
proportion25 = proportion25[,-c(2:4)]
names(proportion25)[2] = "Pfreshwater25"
proportion27$prop_marine = NULL
proportion27 = proportion27[,-c(2:4)]
names(proportion27)[2] = "Pfreshwater27"


# Merge all the information in the same dataframe
heatmappop = merge(proportion5, proportion6, by="Population")
heatmappop = merge(heatmappop, proportion10, by="Population")
heatmappop = merge(heatmappop, proportion11, by="Population")
heatmappop = merge(heatmappop, proportion12, by="Population")
heatmappop = merge(heatmappop, proportion13, by="Population")
heatmappop = merge(heatmappop, proportion16, by="Population")
heatmappop = merge(heatmappop, proportion18, by="Population")
heatmappop = merge(heatmappop, proportion20, by="Population")
heatmappop = merge(heatmappop, proportion22, by="Population")
heatmappop = merge(heatmappop, proportion25, by="Population")
heatmappop = merge(heatmappop, proportion27, by="Population")

# Melt the dataframe
heatmapplot = melt(data=heatmappop, id.vars=c("Population"))

# Label color
color_label_marine = rep("navy", 7)
color_label_freshwater = rep("skyblue1", 20)
color_labels = c(color_label_freshwater, color_label_marine)

# Save the heatmap
png("Heatmap Populations.png", units="in", width=12, height=10, res=900)
ggplot(data=heatmapplot, aes(x=variable, y=Population, fill=value)) +
  geom_tile() +
  xlab("Cluster") +
  labs(fill="P. Freshwater") +
  scale_x_discrete(labels=c("5", "6", "10", "11", "12", "13", "16", "18", "20", "22", "25", "27")) +
  scale_y_discrete(limits = rev(levels(heatmapplot$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  theme(axis.text.y=element_text(color=color_labels))
dev.off()
 
#### Ridgeplots ####

# Label color
color_label_marine = rep("navy", 7)
color_label_freshwater = rep("skyblue1", 20)
color_labels = c(color_label_freshwater, color_label_marine)

# Plots
ridge5 = ggplot(data=finaldata5, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 5 - Chr. 1 (46 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata5$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  coord_cartesian() +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge6 = ggplot(data=finaldata6, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 6 - Chr. 1 (988 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata6$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge10 = ggplot(data=finaldata10, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 10 - Chr. 4 (64 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata10$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge11 = ggplot(data=finaldata11, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 11 - Chr. 4 (326 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata11$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge12 = ggplot(data=finaldata12, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 12 - Chr. 4 (60 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata12$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge13 = ggplot(data=finaldata13, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 13 - Chr. 4 (113 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata13$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge16 = ggplot(data=finaldata16, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 16 - Chr. 8 (34 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata16$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge18 = ggplot(data=finaldata18, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 18 - Chr. 9 (30 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata18$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge20 = ggplot(data=finaldata20, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 20  - Chr. 9 (45 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata20$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge22 = ggplot(data=finaldata22, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 22  - Chr. 11 (517 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata22$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge25 = ggplot(data=finaldata25, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 25  - Chr. 16 (36 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata25$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

ridge27 = ggplot(data=finaldata27, aes(x=Posterior_Freshwater, y=Population, fill=stat(x))) +
  geom_density_ridges_gradient(bandwidth=0.04) +
  geom_point(data=subset(finaldata5, Population %in% c("RUS-LEV-GA", "NOR-BAR-GA", "NOR-SBJ-GA", "GAC-RUS-PRI", "GAC-NOR-KRI", "GAC-SWE-FIS", "ENG-BUT-GA", "ITA-STE")),
             aes(shape=True_Type, alpha=True_Type), size=4) +
  ggtitle("Cluster 27  - Chr. 20 (222 loci)") +
  xlab("Mean Probability of Freshwater") +
  labs(fill="P. Freshwater") +
  scale_y_discrete(limits = rev(levels(finaldata27$Population))) +
  scale_fill_gradientn(colours=c("blue", "yellow", "red")) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_alpha_manual(values=c("Freshwater"=0.5, "Marine"=0.7)) +
  theme_bw() +
  theme(axis.text.y=element_text(color=color_labels),
        plot.title = element_text(hjust=0.5))

png("RidgePlot.png", units="in", width=15, height=20, res=900)
ggarrange(ridge5,
          ridge6 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge10 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge11,
          ridge12 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge13 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge16,
          ridge18 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge20 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge22,
          ridge25 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ridge27 + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.title.y=element_blank()),
          ncol=3, nrow=4, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="right")
dev.off()

#### LM Heatmap ####

# Let's make lm taking the ecotype
C5NORKVAGA = lm(V1~Type, data=subset(pca5, Population=="NOR-KVA-GA" | Type=="Marine"))
C5NORSKFGA = lm(V1~Type, data=subset(pca5, Population=="NOR-SKF-GA" | Type=="Marine"))
C5FINKEVGA = lm(V1~Type, data=subset(pca5, Population=="FIN-KEV-GA" | Type=="Marine"))
C5RUSSLIGA = lm(V1~Type, data=subset(pca5, Population=="RUS-SLI-GA" | Type=="Marine"))
C5NORORRGA = lm(V1~Type, data=subset(pca5, Population=="NOR-ORR-GA" | Type=="Marine"))
C5ENGBUTGA = lm(V1~Type, data=subset(pca5, Population=="ENG-BUT-GA" | Type=="Marine"))
C5GAPORVO = lm(V1~Type, data=subset(pca5, Population=="GA-POR-VO" | Type=="Marine"))
C5GAPORRA = lm(V1~Type, data=subset(pca5, Population=="GA-POR-RA" | Type=="Marine"))
C5GAPORLI = lm(V1~Type, data=subset(pca5, Population=="GA-POR-LI" | Type=="Marine"))
C5GAPORTE = lm(V1~Type, data=subset(pca5, Population=="GA-POR-TE" | Type=="Marine"))
C5GAPORMI = lm(V1~Type, data=subset(pca5, Population=="GA-POR-MI" | Type=="Marine"))
C5GAPORSA = lm(V1~Type, data=subset(pca5, Population=="GA-POR-SA" | Type=="Marine"))
C5MU = lm(V1~Type, data=subset(pca5, Population=="MU" | Type=="Marine"))
C5NN = lm(V1~Type, data=subset(pca5, Population=="NN" | Type=="Marine"))
C5ITASTE = lm(V1~Type, data=subset(pca5, Population=="ITA-STE" | Type=="Marine"))
C5BOSNERGA = lm(V1~Type, data=subset(pca5, Population=="BOS-NER-GA" | Type=="Marine"))
C5MONSKAGA = lm(V1~Type, data=subset(pca5, Population=="MON-SKA-GA" | Type=="Marine"))
C5M = lm(V1~Type, data=subset(pca5, Population=="M" | Type=="Marine"))
C5NB = lm(V1~Type, data=subset(pca5, Population=="NB" | Type=="Marine"))
C5KR = lm(V1~Type, data=subset(pca5, Population=="KR" | Type=="Marine"))
C5BEARPAWLAKE = lm(V1~Type, data=subset(pca5, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C5BOOTLAKE = lm(V1~Type, data=subset(pca5, Population=="BOOT-LAKE" | Type=="Marine"))

C6NORKVAGA = lm(V1~Type, data=subset(pca6, Population=="NOR-KVA-GA" | Type=="Marine"))
C6NORSKFGA = lm(V1~Type, data=subset(pca6, Population=="NOR-SKF-GA" | Type=="Marine"))
C6FINKEVGA = lm(V1~Type, data=subset(pca6, Population=="FIN-KEV-GA" | Type=="Marine"))
C6RUSSLIGA = lm(V1~Type, data=subset(pca6, Population=="RUS-SLI-GA" | Type=="Marine"))
C6NORORRGA = lm(V1~Type, data=subset(pca6, Population=="NOR-ORR-GA" | Type=="Marine"))
C6ENGBUTGA = lm(V1~Type, data=subset(pca6, Population=="ENG-BUT-GA" | Type=="Marine"))
C6GAPORVO = lm(V1~Type, data=subset(pca6, Population=="GA-POR-VO" | Type=="Marine"))
C6GAPORRA = lm(V1~Type, data=subset(pca6, Population=="GA-POR-RA" | Type=="Marine"))
C6GAPORLI = lm(V1~Type, data=subset(pca6, Population=="GA-POR-LI" | Type=="Marine"))
C6GAPORTE = lm(V1~Type, data=subset(pca6, Population=="GA-POR-TE" | Type=="Marine"))
C6GAPORMI = lm(V1~Type, data=subset(pca6, Population=="GA-POR-MI" | Type=="Marine"))
C6GAPORSA = lm(V1~Type, data=subset(pca6, Population=="GA-POR-SA" | Type=="Marine"))
C6MU = lm(V1~Type, data=subset(pca6, Population=="MU" | Type=="Marine"))
C6NN = lm(V1~Type, data=subset(pca6, Population=="NN" | Type=="Marine"))
C6ITASTE = lm(V1~Type, data=subset(pca6, Population=="ITA-STE" | Type=="Marine"))
C6BOSNERGA = lm(V1~Type, data=subset(pca6, Population=="BOS-NER-GA" | Type=="Marine"))
C6MONSKAGA = lm(V1~Type, data=subset(pca6, Population=="MON-SKA-GA" | Type=="Marine"))
C6M = lm(V1~Type, data=subset(pca6, Population=="M" | Type=="Marine"))
C6NB = lm(V1~Type, data=subset(pca6, Population=="NB" | Type=="Marine"))
C6KR = lm(V1~Type, data=subset(pca6, Population=="KR" | Type=="Marine"))
C6BEARPAWLAKE = lm(V1~Type, data=subset(pca6, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C6BOOTLAKE = lm(V1~Type, data=subset(pca6, Population=="BOOT-LAKE" | Type=="Marine"))

C10NORKVAGA = lm(V1~Type, data=subset(pca10, Population=="NOR-KVA-GA" | Type=="Marine"))
C10NORSKFGA = lm(V1~Type, data=subset(pca10, Population=="NOR-SKF-GA" | Type=="Marine"))
C10FINKEVGA = lm(V1~Type, data=subset(pca10, Population=="FIN-KEV-GA" | Type=="Marine"))
C10RUSSLIGA = lm(V1~Type, data=subset(pca10, Population=="RUS-SLI-GA" | Type=="Marine"))
C10NORORRGA = lm(V1~Type, data=subset(pca10, Population=="NOR-ORR-GA" | Type=="Marine"))
C10ENGBUTGA = lm(V1~Type, data=subset(pca10, Population=="ENG-BUT-GA" | Type=="Marine"))
C10GAPORVO = lm(V1~Type, data=subset(pca10, Population=="GA-POR-VO" | Type=="Marine"))
C10GAPORRA = lm(V1~Type, data=subset(pca10, Population=="GA-POR-RA" | Type=="Marine"))
C10GAPORLI = lm(V1~Type, data=subset(pca10, Population=="GA-POR-LI" | Type=="Marine"))
C10GAPORTE = lm(V1~Type, data=subset(pca10, Population=="GA-POR-TE" | Type=="Marine"))
C10GAPORMI = lm(V1~Type, data=subset(pca10, Population=="GA-POR-MI" | Type=="Marine"))
C10GAPORSA = lm(V1~Type, data=subset(pca10, Population=="GA-POR-SA" | Type=="Marine"))
C10MU = lm(V1~Type, data=subset(pca10, Population=="MU" | Type=="Marine"))
C10NN = lm(V1~Type, data=subset(pca10, Population=="NN" | Type=="Marine"))
C10ITASTE = lm(V1~Type, data=subset(pca10, Population=="ITA-STE" | Type=="Marine"))
C10BOSNERGA = lm(V1~Type, data=subset(pca10, Population=="BOS-NER-GA" | Type=="Marine"))
C10MONSKAGA = lm(V1~Type, data=subset(pca10, Population=="MON-SKA-GA" | Type=="Marine"))
C10M = lm(V1~Type, data=subset(pca10, Population=="M" | Type=="Marine"))
C10NB = lm(V1~Type, data=subset(pca10, Population=="NB" | Type=="Marine"))
C10KR = lm(V1~Type, data=subset(pca10, Population=="KR" | Type=="Marine"))
C10BEARPAWLAKE = lm(V1~Type, data=subset(pca10, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C10BOOTLAKE = lm(V1~Type, data=subset(pca10, Population=="BOOT-LAKE" | Type=="Marine"))

C11NORKVAGA = lm(V1~Type, data=subset(pca11, Population=="NOR-KVA-GA" | Type=="Marine"))
C11NORSKFGA = lm(V1~Type, data=subset(pca11, Population=="NOR-SKF-GA" | Type=="Marine"))
C11FINKEVGA = lm(V1~Type, data=subset(pca11, Population=="FIN-KEV-GA" | Type=="Marine"))
C11RUSSLIGA = lm(V1~Type, data=subset(pca11, Population=="RUS-SLI-GA" | Type=="Marine"))
C11NORORRGA = lm(V1~Type, data=subset(pca11, Population=="NOR-ORR-GA" | Type=="Marine"))
C11ENGBUTGA = lm(V1~Type, data=subset(pca11, Population=="ENG-BUT-GA" | Type=="Marine"))
C11GAPORVO = lm(V1~Type, data=subset(pca11, Population=="GA-POR-VO" | Type=="Marine"))
C11GAPORRA = lm(V1~Type, data=subset(pca11, Population=="GA-POR-RA" | Type=="Marine"))
C11GAPORLI = lm(V1~Type, data=subset(pca11, Population=="GA-POR-LI" | Type=="Marine"))
C11GAPORTE = lm(V1~Type, data=subset(pca11, Population=="GA-POR-TE" | Type=="Marine"))
C11GAPORMI = lm(V1~Type, data=subset(pca11, Population=="GA-POR-MI" | Type=="Marine"))
C11GAPORSA = lm(V1~Type, data=subset(pca11, Population=="GA-POR-SA" | Type=="Marine"))
C11MU = lm(V1~Type, data=subset(pca11, Population=="MU" | Type=="Marine"))
C11NN = lm(V1~Type, data=subset(pca11, Population=="NN" | Type=="Marine"))
C11ITASTE = lm(V1~Type, data=subset(pca11, Population=="ITA-STE" | Type=="Marine"))
C11BOSNERGA = lm(V1~Type, data=subset(pca11, Population=="BOS-NER-GA" | Type=="Marine"))
C11MONSKAGA = lm(V1~Type, data=subset(pca11, Population=="MON-SKA-GA" | Type=="Marine"))
C11M = lm(V1~Type, data=subset(pca11, Population=="M" | Type=="Marine"))
C11NB = lm(V1~Type, data=subset(pca11, Population=="NB" | Type=="Marine"))
C11KR = lm(V1~Type, data=subset(pca11, Population=="KR" | Type=="Marine"))
C11BEARPAWLAKE = lm(V1~Type, data=subset(pca11, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C11BOOTLAKE = lm(V1~Type, data=subset(pca11, Population=="BOOT-LAKE" | Type=="Marine"))

C12NORKVAGA = lm(V1~Type, data=subset(pca12, Population=="NOR-KVA-GA" | Type=="Marine"))
C12NORSKFGA = lm(V1~Type, data=subset(pca12, Population=="NOR-SKF-GA" | Type=="Marine"))
C12FINKEVGA = lm(V1~Type, data=subset(pca12, Population=="FIN-KEV-GA" | Type=="Marine"))
C12RUSSLIGA = lm(V1~Type, data=subset(pca12, Population=="RUS-SLI-GA" | Type=="Marine"))
C12NORORRGA = lm(V1~Type, data=subset(pca12, Population=="NOR-ORR-GA" | Type=="Marine"))
C12ENGBUTGA = lm(V1~Type, data=subset(pca12, Population=="ENG-BUT-GA" | Type=="Marine"))
C12GAPORVO = lm(V1~Type, data=subset(pca12, Population=="GA-POR-VO" | Type=="Marine"))
C12GAPORRA = lm(V1~Type, data=subset(pca12, Population=="GA-POR-RA" | Type=="Marine"))
C12GAPORLI = lm(V1~Type, data=subset(pca12, Population=="GA-POR-LI" | Type=="Marine"))
C12GAPORTE = lm(V1~Type, data=subset(pca12, Population=="GA-POR-TE" | Type=="Marine"))
C12GAPORMI = lm(V1~Type, data=subset(pca12, Population=="GA-POR-MI" | Type=="Marine"))
C12GAPORSA = lm(V1~Type, data=subset(pca12, Population=="GA-POR-SA" | Type=="Marine"))
C12MU = lm(V1~Type, data=subset(pca12, Population=="MU" | Type=="Marine"))
C12NN = lm(V1~Type, data=subset(pca12, Population=="NN" | Type=="Marine"))
C12ITASTE = lm(V1~Type, data=subset(pca12, Population=="ITA-STE" | Type=="Marine"))
C12BOSNERGA = lm(V1~Type, data=subset(pca12, Population=="BOS-NER-GA" | Type=="Marine"))
C12MONSKAGA = lm(V1~Type, data=subset(pca12, Population=="MON-SKA-GA" | Type=="Marine"))
C12M = lm(V1~Type, data=subset(pca12, Population=="M" | Type=="Marine"))
C12NB = lm(V1~Type, data=subset(pca12, Population=="NB" | Type=="Marine"))
C12KR = lm(V1~Type, data=subset(pca12, Population=="KR" | Type=="Marine"))
C12BEARPAWLAKE = lm(V1~Type, data=subset(pca12, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C12BOOTLAKE = lm(V1~Type, data=subset(pca12, Population=="BOOT-LAKE" | Type=="Marine"))

C13NORKVAGA = lm(V1~Type, data=subset(pca13, Population=="NOR-KVA-GA" | Type=="Marine"))
C13NORSKFGA = lm(V1~Type, data=subset(pca13, Population=="NOR-SKF-GA" | Type=="Marine"))
C13FINKEVGA = lm(V1~Type, data=subset(pca13, Population=="FIN-KEV-GA" | Type=="Marine"))
C13RUSSLIGA = lm(V1~Type, data=subset(pca13, Population=="RUS-SLI-GA" | Type=="Marine"))
C13NORORRGA = lm(V1~Type, data=subset(pca13, Population=="NOR-ORR-GA" | Type=="Marine"))
C13ENGBUTGA = lm(V1~Type, data=subset(pca13, Population=="ENG-BUT-GA" | Type=="Marine"))
C13GAPORVO = lm(V1~Type, data=subset(pca13, Population=="GA-POR-VO" | Type=="Marine"))
C13GAPORRA = lm(V1~Type, data=subset(pca13, Population=="GA-POR-RA" | Type=="Marine"))
C13GAPORLI = lm(V1~Type, data=subset(pca13, Population=="GA-POR-LI" | Type=="Marine"))
C13GAPORTE = lm(V1~Type, data=subset(pca13, Population=="GA-POR-TE" | Type=="Marine"))
C13GAPORMI = lm(V1~Type, data=subset(pca13, Population=="GA-POR-MI" | Type=="Marine"))
C13GAPORSA = lm(V1~Type, data=subset(pca13, Population=="GA-POR-SA" | Type=="Marine"))
C13MU = lm(V1~Type, data=subset(pca13, Population=="MU" | Type=="Marine"))
C13NN = lm(V1~Type, data=subset(pca13, Population=="NN" | Type=="Marine"))
C13ITASTE = lm(V1~Type, data=subset(pca13, Population=="ITA-STE" | Type=="Marine"))
C13BOSNERGA = lm(V1~Type, data=subset(pca13, Population=="BOS-NER-GA" | Type=="Marine"))
C13MONSKAGA = lm(V1~Type, data=subset(pca13, Population=="MON-SKA-GA" | Type=="Marine"))
C13M = lm(V1~Type, data=subset(pca13, Population=="M" | Type=="Marine"))
C13NB = lm(V1~Type, data=subset(pca13, Population=="NB" | Type=="Marine"))
C13KR = lm(V1~Type, data=subset(pca13, Population=="KR" | Type=="Marine"))
C13BEARPAWLAKE = lm(V1~Type, data=subset(pca13, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C13BOOTLAKE = lm(V1~Type, data=subset(pca13, Population=="BOOT-LAKE" | Type=="Marine"))

C16NORKVAGA = lm(V1~Type, data=subset(pca16, Population=="NOR-KVA-GA" | Type=="Marine"))
C16NORSKFGA = lm(V1~Type, data=subset(pca16, Population=="NOR-SKF-GA" | Type=="Marine"))
C16FINKEVGA = lm(V1~Type, data=subset(pca16, Population=="FIN-KEV-GA" | Type=="Marine"))
C16RUSSLIGA = lm(V1~Type, data=subset(pca16, Population=="RUS-SLI-GA" | Type=="Marine"))
C16NORORRGA = lm(V1~Type, data=subset(pca16, Population=="NOR-ORR-GA" | Type=="Marine"))
C16ENGBUTGA = lm(V1~Type, data=subset(pca16, Population=="ENG-BUT-GA" | Type=="Marine"))
C16GAPORVO = lm(V1~Type, data=subset(pca16, Population=="GA-POR-VO" | Type=="Marine"))
C16GAPORRA = lm(V1~Type, data=subset(pca16, Population=="GA-POR-RA" | Type=="Marine"))
C16GAPORLI = lm(V1~Type, data=subset(pca16, Population=="GA-POR-LI" | Type=="Marine"))
C16GAPORTE = lm(V1~Type, data=subset(pca16, Population=="GA-POR-TE" | Type=="Marine"))
C16GAPORMI = lm(V1~Type, data=subset(pca16, Population=="GA-POR-MI" | Type=="Marine"))
C16GAPORSA = lm(V1~Type, data=subset(pca16, Population=="GA-POR-SA" | Type=="Marine"))
C16MU = lm(V1~Type, data=subset(pca16, Population=="MU" | Type=="Marine"))
C16NN = lm(V1~Type, data=subset(pca16, Population=="NN" | Type=="Marine"))
C16ITASTE = lm(V1~Type, data=subset(pca16, Population=="ITA-STE" | Type=="Marine"))
C16BOSNERGA = lm(V1~Type, data=subset(pca16, Population=="BOS-NER-GA" | Type=="Marine"))
C16MONSKAGA = lm(V1~Type, data=subset(pca16, Population=="MON-SKA-GA" | Type=="Marine"))
C16M = lm(V1~Type, data=subset(pca16, Population=="M" | Type=="Marine"))
C16NB = lm(V1~Type, data=subset(pca16, Population=="NB" | Type=="Marine"))
C16KR = lm(V1~Type, data=subset(pca16, Population=="KR" | Type=="Marine"))
C16BEARPAWLAKE = lm(V1~Type, data=subset(pca16, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C16BOOTLAKE = lm(V1~Type, data=subset(pca16, Population=="BOOT-LAKE" | Type=="Marine"))

C18NORKVAGA = lm(V1~Type, data=subset(pca18, Population=="NOR-KVA-GA" | Type=="Marine"))
C18NORSKFGA = lm(V1~Type, data=subset(pca18, Population=="NOR-SKF-GA" | Type=="Marine"))
C18FINKEVGA = lm(V1~Type, data=subset(pca18, Population=="FIN-KEV-GA" | Type=="Marine"))
C18RUSSLIGA = lm(V1~Type, data=subset(pca18, Population=="RUS-SLI-GA" | Type=="Marine"))
C18NORORRGA = lm(V1~Type, data=subset(pca18, Population=="NOR-ORR-GA" | Type=="Marine"))
C18ENGBUTGA = lm(V1~Type, data=subset(pca18, Population=="ENG-BUT-GA" | Type=="Marine"))
C18GAPORVO = lm(V1~Type, data=subset(pca18, Population=="GA-POR-VO" | Type=="Marine"))
C18GAPORRA = lm(V1~Type, data=subset(pca18, Population=="GA-POR-RA" | Type=="Marine"))
C18GAPORLI = lm(V1~Type, data=subset(pca18, Population=="GA-POR-LI" | Type=="Marine"))
C18GAPORTE = lm(V1~Type, data=subset(pca18, Population=="GA-POR-TE" | Type=="Marine"))
C18GAPORMI = lm(V1~Type, data=subset(pca18, Population=="GA-POR-MI" | Type=="Marine"))
C18GAPORSA = lm(V1~Type, data=subset(pca18, Population=="GA-POR-SA" | Type=="Marine"))
C18MU = lm(V1~Type, data=subset(pca18, Population=="MU" | Type=="Marine"))
C18NN = lm(V1~Type, data=subset(pca18, Population=="NN" | Type=="Marine"))
C18ITASTE = lm(V1~Type, data=subset(pca18, Population=="ITA-STE" | Type=="Marine"))
C18BOSNERGA = lm(V1~Type, data=subset(pca18, Population=="BOS-NER-GA" | Type=="Marine"))
C18MONSKAGA = lm(V1~Type, data=subset(pca18, Population=="MON-SKA-GA" | Type=="Marine"))
C18M = lm(V1~Type, data=subset(pca18, Population=="M" | Type=="Marine"))
C18NB = lm(V1~Type, data=subset(pca18, Population=="NB" | Type=="Marine"))
C18KR = lm(V1~Type, data=subset(pca18, Population=="KR" | Type=="Marine"))
C18BEARPAWLAKE = lm(V1~Type, data=subset(pca18, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C18BOOTLAKE = lm(V1~Type, data=subset(pca18, Population=="BOOT-LAKE" | Type=="Marine"))

C20NORKVAGA = lm(V1~Type, data=subset(pca20, Population=="NOR-KVA-GA" | Type=="Marine"))
C20NORSKFGA = lm(V1~Type, data=subset(pca20, Population=="NOR-SKF-GA" | Type=="Marine"))
C20FINKEVGA = lm(V1~Type, data=subset(pca20, Population=="FIN-KEV-GA" | Type=="Marine"))
C20RUSSLIGA = lm(V1~Type, data=subset(pca20, Population=="RUS-SLI-GA" | Type=="Marine"))
C20NORORRGA = lm(V1~Type, data=subset(pca20, Population=="NOR-ORR-GA" | Type=="Marine"))
C20ENGBUTGA = lm(V1~Type, data=subset(pca20, Population=="ENG-BUT-GA" | Type=="Marine"))
C20GAPORVO = lm(V1~Type, data=subset(pca20, Population=="GA-POR-VO" | Type=="Marine"))
C20GAPORRA = lm(V1~Type, data=subset(pca20, Population=="GA-POR-RA" | Type=="Marine"))
C20GAPORLI = lm(V1~Type, data=subset(pca20, Population=="GA-POR-LI" | Type=="Marine"))
C20GAPORTE = lm(V1~Type, data=subset(pca20, Population=="GA-POR-TE" | Type=="Marine"))
C20GAPORMI = lm(V1~Type, data=subset(pca20, Population=="GA-POR-MI" | Type=="Marine"))
C20GAPORSA = lm(V1~Type, data=subset(pca20, Population=="GA-POR-SA" | Type=="Marine"))
C20MU = lm(V1~Type, data=subset(pca20, Population=="MU" | Type=="Marine"))
C20NN = lm(V1~Type, data=subset(pca20, Population=="NN" | Type=="Marine"))
C20ITASTE = lm(V1~Type, data=subset(pca20, Population=="ITA-STE" | Type=="Marine"))
C20BOSNERGA = lm(V1~Type, data=subset(pca20, Population=="BOS-NER-GA" | Type=="Marine"))
C20MONSKAGA = lm(V1~Type, data=subset(pca20, Population=="MON-SKA-GA" | Type=="Marine"))
C20M = lm(V1~Type, data=subset(pca20, Population=="M" | Type=="Marine"))
C20NB = lm(V1~Type, data=subset(pca20, Population=="NB" | Type=="Marine"))
C20KR = lm(V1~Type, data=subset(pca20, Population=="KR" | Type=="Marine"))
C20BEARPAWLAKE = lm(V1~Type, data=subset(pca20, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C20BOOTLAKE = lm(V1~Type, data=subset(pca20, Population=="BOOT-LAKE" | Type=="Marine"))

C22NORKVAGA = lm(V1~Type, data=subset(pca22, Population=="NOR-KVA-GA" | Type=="Marine"))
C22NORSKFGA = lm(V1~Type, data=subset(pca22, Population=="NOR-SKF-GA" | Type=="Marine"))
C22FINKEVGA = lm(V1~Type, data=subset(pca22, Population=="FIN-KEV-GA" | Type=="Marine"))
C22RUSSLIGA = lm(V1~Type, data=subset(pca22, Population=="RUS-SLI-GA" | Type=="Marine"))
C22NORORRGA = lm(V1~Type, data=subset(pca22, Population=="NOR-ORR-GA" | Type=="Marine"))
C22ENGBUTGA = lm(V1~Type, data=subset(pca22, Population=="ENG-BUT-GA" | Type=="Marine"))
C22GAPORVO = lm(V1~Type, data=subset(pca22, Population=="GA-POR-VO" | Type=="Marine"))
C22GAPORRA = lm(V1~Type, data=subset(pca22, Population=="GA-POR-RA" | Type=="Marine"))
C22GAPORLI = lm(V1~Type, data=subset(pca22, Population=="GA-POR-LI" | Type=="Marine"))
C22GAPORTE = lm(V1~Type, data=subset(pca22, Population=="GA-POR-TE" | Type=="Marine"))
C22GAPORMI = lm(V1~Type, data=subset(pca22, Population=="GA-POR-MI" | Type=="Marine"))
C22GAPORSA = lm(V1~Type, data=subset(pca22, Population=="GA-POR-SA" | Type=="Marine"))
C22MU = lm(V1~Type, data=subset(pca22, Population=="MU" | Type=="Marine"))
C22NN = lm(V1~Type, data=subset(pca22, Population=="NN" | Type=="Marine"))
C22ITASTE = lm(V1~Type, data=subset(pca22, Population=="ITA-STE" | Type=="Marine"))
C22BOSNERGA = lm(V1~Type, data=subset(pca22, Population=="BOS-NER-GA" | Type=="Marine"))
C22MONSKAGA = lm(V1~Type, data=subset(pca22, Population=="MON-SKA-GA" | Type=="Marine"))
C22M = lm(V1~Type, data=subset(pca22, Population=="M" | Type=="Marine"))
C22NB = lm(V1~Type, data=subset(pca22, Population=="NB" | Type=="Marine"))
C22KR = lm(V1~Type, data=subset(pca22, Population=="KR" | Type=="Marine"))
C22BEARPAWLAKE = lm(V1~Type, data=subset(pca22, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C22BOOTLAKE = lm(V1~Type, data=subset(pca22, Population=="BOOT-LAKE" | Type=="Marine"))

C25NORKVAGA = lm(V1~Type, data=subset(pca25, Population=="NOR-KVA-GA" | Type=="Marine"))
C25NORSKFGA = lm(V1~Type, data=subset(pca25, Population=="NOR-SKF-GA" | Type=="Marine"))
C25FINKEVGA = lm(V1~Type, data=subset(pca25, Population=="FIN-KEV-GA" | Type=="Marine"))
C25RUSSLIGA = lm(V1~Type, data=subset(pca25, Population=="RUS-SLI-GA" | Type=="Marine"))
C25NORORRGA = lm(V1~Type, data=subset(pca25, Population=="NOR-ORR-GA" | Type=="Marine"))
C25ENGBUTGA = lm(V1~Type, data=subset(pca25, Population=="ENG-BUT-GA" | Type=="Marine"))
C25GAPORVO = lm(V1~Type, data=subset(pca25, Population=="GA-POR-VO" | Type=="Marine"))
C25GAPORRA = lm(V1~Type, data=subset(pca25, Population=="GA-POR-RA" | Type=="Marine"))
C25GAPORLI = lm(V1~Type, data=subset(pca25, Population=="GA-POR-LI" | Type=="Marine"))
C25GAPORTE = lm(V1~Type, data=subset(pca25, Population=="GA-POR-TE" | Type=="Marine"))
C25GAPORMI = lm(V1~Type, data=subset(pca25, Population=="GA-POR-MI" | Type=="Marine"))
C25GAPORSA = lm(V1~Type, data=subset(pca25, Population=="GA-POR-SA" | Type=="Marine"))
C25MU = lm(V1~Type, data=subset(pca25, Population=="MU" | Type=="Marine"))
C25NN = lm(V1~Type, data=subset(pca25, Population=="NN" | Type=="Marine"))
C25ITASTE = lm(V1~Type, data=subset(pca25, Population=="ITA-STE" | Type=="Marine"))
C25BOSNERGA = lm(V1~Type, data=subset(pca25, Population=="BOS-NER-GA" | Type=="Marine"))
C25MONSKAGA = lm(V1~Type, data=subset(pca25, Population=="MON-SKA-GA" | Type=="Marine"))
C25M = lm(V1~Type, data=subset(pca25, Population=="M" | Type=="Marine"))
C25NB = lm(V1~Type, data=subset(pca25, Population=="NB" | Type=="Marine"))
C25KR = lm(V1~Type, data=subset(pca25, Population=="KR" | Type=="Marine"))
C25BEARPAWLAKE = lm(V1~Type, data=subset(pca25, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C25BOOTLAKE = lm(V1~Type, data=subset(pca25, Population=="BOOT-LAKE" | Type=="Marine"))

C27NORKVAGA = lm(V1~Type, data=subset(pca27, Population=="NOR-KVA-GA" | Type=="Marine"))
C27NORSKFGA = lm(V1~Type, data=subset(pca27, Population=="NOR-SKF-GA" | Type=="Marine"))
C27FINKEVGA = lm(V1~Type, data=subset(pca27, Population=="FIN-KEV-GA" | Type=="Marine"))
C27RUSSLIGA = lm(V1~Type, data=subset(pca27, Population=="RUS-SLI-GA" | Type=="Marine"))
C27NORORRGA = lm(V1~Type, data=subset(pca27, Population=="NOR-ORR-GA" | Type=="Marine"))
C27ENGBUTGA = lm(V1~Type, data=subset(pca27, Population=="ENG-BUT-GA" | Type=="Marine"))
C27GAPORVO = lm(V1~Type, data=subset(pca27, Population=="GA-POR-VO" | Type=="Marine"))
C27GAPORRA = lm(V1~Type, data=subset(pca27, Population=="GA-POR-RA" | Type=="Marine"))
C27GAPORLI = lm(V1~Type, data=subset(pca27, Population=="GA-POR-LI" | Type=="Marine"))
C27GAPORTE = lm(V1~Type, data=subset(pca27, Population=="GA-POR-TE" | Type=="Marine"))
C27GAPORMI = lm(V1~Type, data=subset(pca27, Population=="GA-POR-MI" | Type=="Marine"))
C27GAPORSA = lm(V1~Type, data=subset(pca27, Population=="GA-POR-SA" | Type=="Marine"))
C27MU = lm(V1~Type, data=subset(pca27, Population=="MU" | Type=="Marine"))
C27NN = lm(V1~Type, data=subset(pca27, Population=="NN" | Type=="Marine"))
C27ITASTE = lm(V1~Type, data=subset(pca27, Population=="ITA-STE" | Type=="Marine"))
C27BOSNERGA = lm(V1~Type, data=subset(pca27, Population=="BOS-NER-GA" | Type=="Marine"))
C27MONSKAGA = lm(V1~Type, data=subset(pca27, Population=="MON-SKA-GA" | Type=="Marine"))
C27M = lm(V1~Type, data=subset(pca27, Population=="M" | Type=="Marine"))
C27NB = lm(V1~Type, data=subset(pca27, Population=="NB" | Type=="Marine"))
C27KR = lm(V1~Type, data=subset(pca27, Population=="KR" | Type=="Marine"))
C27BEARPAWLAKE = lm(V1~Type, data=subset(pca27, Population=="BEAR-PAW-LAKE" | Type=="Marine"))
C27BOOTLAKE = lm(V1~Type, data=subset(pca27, Population=="BOOT-LAKE" | Type=="Marine"))

# Start a final dataframe
finaldatalm = data.frame(pca5$Population, pca5$Type)
finaldatalm = unique(finaldatalm)
names(finaldatalm) = c("Population", "Type")
finaldatalm = finaldatalm[!(finaldatalm$Type=="Marine"),]

# Add p-value information on a vector for each cluster
cluster5 = c(anova(C5BEARPAWLAKE)$'Pr(>F)'[1], anova(C5BOOTLAKE)$'Pr(>F)'[1], anova(C5BOSNERGA)$'Pr(>F)'[1], anova(C5ENGBUTGA)$'Pr(>F)'[1], anova(C5FINKEVGA)$'Pr(>F)'[1], anova(C5GAPORLI)$'Pr(>F)'[1], anova(C5GAPORMI)$'Pr(>F)'[1], anova(C5GAPORRA)$'Pr(>F)'[1], anova(C5GAPORSA)$'Pr(>F)'[1], anova(C5GAPORTE)$'Pr(>F)'[1], anova(C5GAPORVO)$'Pr(>F)'[1], anova(C5ITASTE)$'Pr(>F)'[1], anova(C5KR)$'Pr(>F)'[1], anova(C5M)$'Pr(>F)'[1], anova(C5MONSKAGA)$'Pr(>F)'[1], anova(C5MU)$'Pr(>F)'[1], anova(C5NB)$'Pr(>F)'[1], anova(C5NN)$'Pr(>F)'[1], anova(C5NORKVAGA)$'Pr(>F)'[1], anova(C5NORORRGA)$'Pr(>F)'[1],anova(C5NORSKFGA)$'Pr(>F)'[1], anova(C5RUSSLIGA)$'Pr(>F)'[1])  
cluster6 = c(anova(C6BEARPAWLAKE)$'Pr(>F)'[1], anova(C6BOOTLAKE)$'Pr(>F)'[1], anova(C6BOSNERGA)$'Pr(>F)'[1], anova(C6ENGBUTGA)$'Pr(>F)'[1], anova(C6FINKEVGA)$'Pr(>F)'[1], anova(C6GAPORLI)$'Pr(>F)'[1], anova(C6GAPORMI)$'Pr(>F)'[1], anova(C6GAPORRA)$'Pr(>F)'[1], anova(C6GAPORSA)$'Pr(>F)'[1], anova(C6GAPORTE)$'Pr(>F)'[1], anova(C6GAPORVO)$'Pr(>F)'[1], anova(C6ITASTE)$'Pr(>F)'[1], anova(C6KR)$'Pr(>F)'[1], anova(C6M)$'Pr(>F)'[1], anova(C6MONSKAGA)$'Pr(>F)'[1], anova(C6MU)$'Pr(>F)'[1], anova(C6NB)$'Pr(>F)'[1], anova(C6NN)$'Pr(>F)'[1], anova(C6NORKVAGA)$'Pr(>F)'[1], anova(C6NORORRGA)$'Pr(>F)'[1],anova(C6NORSKFGA)$'Pr(>F)'[1], anova(C6RUSSLIGA)$'Pr(>F)'[1])  
cluster10 = c(anova(C10BEARPAWLAKE)$'Pr(>F)'[1], anova(C10BOOTLAKE)$'Pr(>F)'[1], anova(C10BOSNERGA)$'Pr(>F)'[1], anova(C10ENGBUTGA)$'Pr(>F)'[1], anova(C10FINKEVGA)$'Pr(>F)'[1], anova(C10GAPORLI)$'Pr(>F)'[1], anova(C10GAPORMI)$'Pr(>F)'[1], anova(C10GAPORRA)$'Pr(>F)'[1], anova(C10GAPORSA)$'Pr(>F)'[1], anova(C10GAPORTE)$'Pr(>F)'[1], anova(C10GAPORVO)$'Pr(>F)'[1], anova(C10ITASTE)$'Pr(>F)'[1], anova(C10KR)$'Pr(>F)'[1], anova(C10M)$'Pr(>F)'[1], anova(C10MONSKAGA)$'Pr(>F)'[1], anova(C10MU)$'Pr(>F)'[1], anova(C10NB)$'Pr(>F)'[1], anova(C10NN)$'Pr(>F)'[1], anova(C10NORKVAGA)$'Pr(>F)'[1], anova(C10NORORRGA)$'Pr(>F)'[1],anova(C10NORSKFGA)$'Pr(>F)'[1], anova(C10RUSSLIGA)$'Pr(>F)'[1])  
cluster11 = c(anova(C11BEARPAWLAKE)$'Pr(>F)'[1], anova(C11BOOTLAKE)$'Pr(>F)'[1], anova(C11BOSNERGA)$'Pr(>F)'[1], anova(C11ENGBUTGA)$'Pr(>F)'[1], anova(C11FINKEVGA)$'Pr(>F)'[1], anova(C11GAPORLI)$'Pr(>F)'[1], anova(C11GAPORMI)$'Pr(>F)'[1], anova(C11GAPORRA)$'Pr(>F)'[1], anova(C11GAPORSA)$'Pr(>F)'[1], anova(C11GAPORTE)$'Pr(>F)'[1], anova(C11GAPORVO)$'Pr(>F)'[1], anova(C11ITASTE)$'Pr(>F)'[1], anova(C11KR)$'Pr(>F)'[1], anova(C11M)$'Pr(>F)'[1], anova(C11MONSKAGA)$'Pr(>F)'[1], anova(C11MU)$'Pr(>F)'[1], anova(C11NB)$'Pr(>F)'[1], anova(C11NN)$'Pr(>F)'[1], anova(C11NORKVAGA)$'Pr(>F)'[1], anova(C11NORORRGA)$'Pr(>F)'[1],anova(C11NORSKFGA)$'Pr(>F)'[1], anova(C11RUSSLIGA)$'Pr(>F)'[1])  
cluster12 = c(anova(C12BEARPAWLAKE)$'Pr(>F)'[1], anova(C12BOOTLAKE)$'Pr(>F)'[1], anova(C12BOSNERGA)$'Pr(>F)'[1], anova(C12ENGBUTGA)$'Pr(>F)'[1], anova(C12FINKEVGA)$'Pr(>F)'[1], anova(C12GAPORLI)$'Pr(>F)'[1], anova(C12GAPORMI)$'Pr(>F)'[1], anova(C12GAPORRA)$'Pr(>F)'[1], anova(C12GAPORSA)$'Pr(>F)'[1], anova(C12GAPORTE)$'Pr(>F)'[1], anova(C12GAPORVO)$'Pr(>F)'[1], anova(C12ITASTE)$'Pr(>F)'[1], anova(C12KR)$'Pr(>F)'[1], anova(C12M)$'Pr(>F)'[1], anova(C12MONSKAGA)$'Pr(>F)'[1], anova(C12MU)$'Pr(>F)'[1], anova(C12NB)$'Pr(>F)'[1], anova(C12NN)$'Pr(>F)'[1], anova(C12NORKVAGA)$'Pr(>F)'[1], anova(C12NORORRGA)$'Pr(>F)'[1],anova(C12NORSKFGA)$'Pr(>F)'[1], anova(C12RUSSLIGA)$'Pr(>F)'[1])  
cluster13 = c(anova(C13BEARPAWLAKE)$'Pr(>F)'[1], anova(C13BOOTLAKE)$'Pr(>F)'[1], anova(C13BOSNERGA)$'Pr(>F)'[1], anova(C13ENGBUTGA)$'Pr(>F)'[1], anova(C13FINKEVGA)$'Pr(>F)'[1], anova(C13GAPORLI)$'Pr(>F)'[1], anova(C13GAPORMI)$'Pr(>F)'[1], anova(C13GAPORRA)$'Pr(>F)'[1], anova(C13GAPORSA)$'Pr(>F)'[1], anova(C13GAPORTE)$'Pr(>F)'[1], anova(C13GAPORVO)$'Pr(>F)'[1], anova(C13ITASTE)$'Pr(>F)'[1], anova(C13KR)$'Pr(>F)'[1], anova(C13M)$'Pr(>F)'[1], anova(C13MONSKAGA)$'Pr(>F)'[1], anova(C13MU)$'Pr(>F)'[1], anova(C13NB)$'Pr(>F)'[1], anova(C13NN)$'Pr(>F)'[1], anova(C13NORKVAGA)$'Pr(>F)'[1], anova(C13NORORRGA)$'Pr(>F)'[1],anova(C13NORSKFGA)$'Pr(>F)'[1], anova(C13RUSSLIGA)$'Pr(>F)'[1])  
cluster16 = c(anova(C16BEARPAWLAKE)$'Pr(>F)'[1], anova(C16BOOTLAKE)$'Pr(>F)'[1], anova(C16BOSNERGA)$'Pr(>F)'[1], anova(C16ENGBUTGA)$'Pr(>F)'[1], anova(C16FINKEVGA)$'Pr(>F)'[1], anova(C16GAPORLI)$'Pr(>F)'[1], anova(C16GAPORMI)$'Pr(>F)'[1], anova(C16GAPORRA)$'Pr(>F)'[1], anova(C16GAPORSA)$'Pr(>F)'[1], anova(C16GAPORTE)$'Pr(>F)'[1], anova(C16GAPORVO)$'Pr(>F)'[1], anova(C16ITASTE)$'Pr(>F)'[1], anova(C16KR)$'Pr(>F)'[1], anova(C16M)$'Pr(>F)'[1], anova(C16MONSKAGA)$'Pr(>F)'[1], anova(C16MU)$'Pr(>F)'[1], anova(C16NB)$'Pr(>F)'[1], anova(C16NN)$'Pr(>F)'[1], anova(C16NORKVAGA)$'Pr(>F)'[1], anova(C16NORORRGA)$'Pr(>F)'[1],anova(C16NORSKFGA)$'Pr(>F)'[1], anova(C16RUSSLIGA)$'Pr(>F)'[1])  
cluster18 = c(anova(C18BEARPAWLAKE)$'Pr(>F)'[1], anova(C18BOOTLAKE)$'Pr(>F)'[1], anova(C18BOSNERGA)$'Pr(>F)'[1], anova(C18ENGBUTGA)$'Pr(>F)'[1], anova(C18FINKEVGA)$'Pr(>F)'[1], anova(C18GAPORLI)$'Pr(>F)'[1], anova(C18GAPORMI)$'Pr(>F)'[1], anova(C18GAPORRA)$'Pr(>F)'[1], anova(C18GAPORSA)$'Pr(>F)'[1], anova(C18GAPORTE)$'Pr(>F)'[1], anova(C18GAPORVO)$'Pr(>F)'[1], anova(C18ITASTE)$'Pr(>F)'[1], anova(C18KR)$'Pr(>F)'[1], anova(C18M)$'Pr(>F)'[1], anova(C18MONSKAGA)$'Pr(>F)'[1], anova(C18MU)$'Pr(>F)'[1], anova(C18NB)$'Pr(>F)'[1], anova(C18NN)$'Pr(>F)'[1], anova(C18NORKVAGA)$'Pr(>F)'[1], anova(C18NORORRGA)$'Pr(>F)'[1],anova(C18NORSKFGA)$'Pr(>F)'[1], anova(C18RUSSLIGA)$'Pr(>F)'[1])  
cluster20 = c(anova(C20BEARPAWLAKE)$'Pr(>F)'[1], anova(C20BOOTLAKE)$'Pr(>F)'[1], anova(C20BOSNERGA)$'Pr(>F)'[1], anova(C20ENGBUTGA)$'Pr(>F)'[1], anova(C20FINKEVGA)$'Pr(>F)'[1], anova(C20GAPORLI)$'Pr(>F)'[1], anova(C20GAPORMI)$'Pr(>F)'[1], anova(C20GAPORRA)$'Pr(>F)'[1], anova(C20GAPORSA)$'Pr(>F)'[1], anova(C20GAPORTE)$'Pr(>F)'[1], anova(C20GAPORVO)$'Pr(>F)'[1], anova(C20ITASTE)$'Pr(>F)'[1], anova(C20KR)$'Pr(>F)'[1], anova(C20M)$'Pr(>F)'[1], anova(C20MONSKAGA)$'Pr(>F)'[1], anova(C20MU)$'Pr(>F)'[1], anova(C20NB)$'Pr(>F)'[1], anova(C20NN)$'Pr(>F)'[1], anova(C20NORKVAGA)$'Pr(>F)'[1], anova(C20NORORRGA)$'Pr(>F)'[1],anova(C20NORSKFGA)$'Pr(>F)'[1], anova(C20RUSSLIGA)$'Pr(>F)'[1])  
cluster22 = c(anova(C22BEARPAWLAKE)$'Pr(>F)'[1], anova(C22BOOTLAKE)$'Pr(>F)'[1], anova(C22BOSNERGA)$'Pr(>F)'[1], anova(C22ENGBUTGA)$'Pr(>F)'[1], anova(C22FINKEVGA)$'Pr(>F)'[1], anova(C22GAPORLI)$'Pr(>F)'[1], anova(C22GAPORMI)$'Pr(>F)'[1], anova(C22GAPORRA)$'Pr(>F)'[1], anova(C22GAPORSA)$'Pr(>F)'[1], anova(C22GAPORTE)$'Pr(>F)'[1], anova(C22GAPORVO)$'Pr(>F)'[1], anova(C22ITASTE)$'Pr(>F)'[1], anova(C22KR)$'Pr(>F)'[1], anova(C22M)$'Pr(>F)'[1], anova(C22MONSKAGA)$'Pr(>F)'[1], anova(C22MU)$'Pr(>F)'[1], anova(C22NB)$'Pr(>F)'[1], anova(C22NN)$'Pr(>F)'[1], anova(C22NORKVAGA)$'Pr(>F)'[1], anova(C22NORORRGA)$'Pr(>F)'[1],anova(C22NORSKFGA)$'Pr(>F)'[1], anova(C22RUSSLIGA)$'Pr(>F)'[1])  
cluster25 = c(anova(C25BEARPAWLAKE)$'Pr(>F)'[1], anova(C25BOOTLAKE)$'Pr(>F)'[1], anova(C25BOSNERGA)$'Pr(>F)'[1], anova(C25ENGBUTGA)$'Pr(>F)'[1], anova(C25FINKEVGA)$'Pr(>F)'[1], anova(C25GAPORLI)$'Pr(>F)'[1], anova(C25GAPORMI)$'Pr(>F)'[1], anova(C25GAPORRA)$'Pr(>F)'[1], anova(C25GAPORSA)$'Pr(>F)'[1], anova(C25GAPORTE)$'Pr(>F)'[1], anova(C25GAPORVO)$'Pr(>F)'[1], anova(C25ITASTE)$'Pr(>F)'[1], anova(C25KR)$'Pr(>F)'[1], anova(C25M)$'Pr(>F)'[1], anova(C25MONSKAGA)$'Pr(>F)'[1], anova(C25MU)$'Pr(>F)'[1], anova(C25NB)$'Pr(>F)'[1], anova(C25NN)$'Pr(>F)'[1], anova(C25NORKVAGA)$'Pr(>F)'[1], anova(C25NORORRGA)$'Pr(>F)'[1],anova(C25NORSKFGA)$'Pr(>F)'[1], anova(C25RUSSLIGA)$'Pr(>F)'[1])  
cluster27 = c(anova(C27BEARPAWLAKE)$'Pr(>F)'[1], anova(C27BOOTLAKE)$'Pr(>F)'[1], anova(C27BOSNERGA)$'Pr(>F)'[1], anova(C27ENGBUTGA)$'Pr(>F)'[1], anova(C27FINKEVGA)$'Pr(>F)'[1], anova(C27GAPORLI)$'Pr(>F)'[1], anova(C27GAPORMI)$'Pr(>F)'[1], anova(C27GAPORRA)$'Pr(>F)'[1], anova(C27GAPORSA)$'Pr(>F)'[1], anova(C27GAPORTE)$'Pr(>F)'[1], anova(C27GAPORVO)$'Pr(>F)'[1], anova(C27ITASTE)$'Pr(>F)'[1], anova(C27KR)$'Pr(>F)'[1], anova(C27M)$'Pr(>F)'[1], anova(C27MONSKAGA)$'Pr(>F)'[1], anova(C27MU)$'Pr(>F)'[1], anova(C27NB)$'Pr(>F)'[1], anova(C27NN)$'Pr(>F)'[1], anova(C27NORKVAGA)$'Pr(>F)'[1], anova(C27NORORRGA)$'Pr(>F)'[1],anova(C27NORSKFGA)$'Pr(>F)'[1], anova(C27RUSSLIGA)$'Pr(>F)'[1])  

# Add vectors to dataframe
finaldatalm$cluster5 = cluster5
finaldatalm$cluster6 = cluster6
finaldatalm$cluster10 = cluster10
finaldatalm$cluster11 = cluster11
finaldatalm$cluster12 = cluster12
finaldatalm$cluster13 = cluster13
finaldatalm$cluster16 = cluster16
finaldatalm$cluster18 = cluster18
finaldatalm$cluster20 = cluster20
finaldatalm$cluster22 = cluster22
finaldatalm$cluster25 = cluster25
finaldatalm$cluster27 = cluster27

# Melt the dataframe
finaldatalmplot = melt(data=finaldatalm, id.vars=c("Population", "Type"))

# Label color
color_label_freshwater = rep("skyblue1", 22)

# Order populations
finaldatalmplot$Population = factor(finaldatalmplot$Population, levels=c("BEAR-PAW-LAKE","BOOT-LAKE","NOR-KVA-GA","NOR-SKF-GA", "FIN-KEV-GA", "RUS-SLI-GA", "NOR-ORR-GA", "ENG-BUT-GA", "GA-POR-VO", "GA-POR-RA", "GA-POR-LI", "GA-POR-TE", "GA-POR-MI", "GA-POR-SA", "MU", "NN", "ITA-STE", "BOS-NER-GA", "MON-SKA-GA", "M", "NB", "KR")) 

# Save the heatmap
png("Heatmap Populations LM.png", units="in", width=12, height=10, res=900)
ggplot(data=finaldatalmplot, aes(x=variable, y=Population, fill=value)) +
  geom_tile() +
  xlab("Cluster") +
  labs(fill="P-Value") +
  scale_x_discrete(labels=c("5", "6", "10", "11", "12", "13", "16", "18", "20", "22", "25", "27")) +
  scale_y_discrete(limits = rev(levels(finaldatalmplot$Population))) +
  scale_fill_gradientn(colors=c("red","yellow","blue")) +
  theme(axis.text.y=element_text(color=color_label_freshwater))
dev.off()




