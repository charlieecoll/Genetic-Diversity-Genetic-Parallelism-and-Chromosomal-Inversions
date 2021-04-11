##################################################################
##        CHROMOSOMAL INVERSIONS ON PARALLELISM CLUSTERS        ##
##################################################################

## Carla Coll Costa, C3. April 2021.
## This script expects information of covariance matrices calculated with PCAngsd and the beagle files
## used to calculate the covariances matrices. From the beagle files it calculates the heterozygosity
## of the regions analysed and from the covariance matrices it calculates a PCA. Then it plots the 
## heterozygosity against the 1st PC of the PCA in order to see the chromosomal inversions of the regions
## analysed. The plots are saved automatically. Importantly, the script also expects information on each 
## of the populations to colour the plots according to different characteristics: habitat
## type and geographic region.

#### Prepare the data for plotting ####

# Load the required libraries
library(ggplot2)
library(gridExtra)
library(ggpubr)
library(snpStats)

# Set working directory
setwd("<Working_Directory>")

# Import covariance matrices
cov6_inv1 = as.matrix(read.table("<Input_Covariance_Matrix>"))
cov19_inv9 = as.matrix(read.table("<Input_Covariance_Matrix>"))
cov22_inv11 = as.matrix(read.table("<Input_Covariance_Matrix>"))
cov29_inv21 = as.matrix(read.table("<Input_Covariance_Matrix>"))

# Import individuals in the order of the bam files when calculating beagle file
individuals = read.table("<File_with_Individual_Names>")

# Calculate eigen data
eigen_data6_inv1 = eigen(cov6_inv1)
eigen_data19_inv9 = eigen(cov19_inv9)
eigen_data22_inv11 = eigen(cov22_inv11)
eigen_data29_inv21 = eigen(cov29_inv21)

# Transform eigenvectors into a data frame
eigenvec6_inv1 = as.data.frame(eigen_data6_inv1$vectors)
eigenvec19_inv9 = as.data.frame(eigen_data19_inv9$vectors)
eigenvec22_inv11 = as.data.frame(eigen_data22_inv11$vectors)
eigenvec29_inv21 = as.data.frame(eigen_data29_inv21$vectors)

# Add info about the individuals and populations in the data frame
eigenvec6_inv1 = cbind(individuals, eigenvec6_inv1)
eigenvec19_inv9 = cbind(individuals, eigenvec19_inv9)
eigenvec22_inv11 = cbind(individuals, eigenvec22_inv11)
eigenvec29_inv21 = cbind(individuals, eigenvec29_inv21)

# Change column name
names(eigenvec6_inv1)[1] = "Individual"
names(eigenvec19_inv9)[1] = "Individual"
names(eigenvec22_inv11)[1] = "Individual"
names(eigenvec29_inv21)[1] = "Individual"

# Reorder by name of individual
eigenvec6_inv1 = eigenvec6_inv1[order(eigenvec6_inv1$Individual),]
eigenvec19_inv9 = eigenvec19_inv9[order(eigenvec19_inv9$Individual),]
eigenvec22_inv11 = eigenvec22_inv11[order(eigenvec22_inv11$Individual),]
eigenvec29_inv21 = eigenvec29_inv21[order(eigenvec29_inv21$Individual),]

# Import information on the individuals
individuals_information = read.table("<File_with_Information_on_Individuals>", header=T)

# Merge the two dataframes
pca6_inv1 = merge(individuals_information, eigenvec6_inv1, by="Individual")
pca19_inv9 = merge(individuals_information, eigenvec19_inv9, by="Individual")
pca22_inv11 = merge(individuals_information, eigenvec22_inv11, by="Individual")
pca29_inv21 = merge(individuals_information, eigenvec29_inv21, by="Individual")

# Keep only the data of interest
pca6_inv1 = data.frame(pca6_inv1$Individual, pca6_inv1$Population, pca6_inv1$Type, pca6_inv1$Concrete_Location, pca6_inv1$V1, pca6_inv1$V2)
pca19_inv9 = data.frame(pca19_inv9$Individual, pca19_inv9$Population, pca19_inv9$Type, pca19_inv9$Concrete_Location, pca19_inv9$V1, pca19_inv9$V2)
pca22_inv11 = data.frame(pca22_inv11$Individual, pca22_inv11$Population, pca22_inv11$Type, pca22_inv11$Concrete_Location, pca22_inv11$V1, pca22_inv11$V2)
pca29_inv21 = data.frame(pca29_inv21$Individual, pca29_inv21$Population, pca29_inv21$Type, pca29_inv21$Concrete_Location, pca29_inv21$V1, pca29_inv21$V2)

# Change column name
names(pca6_inv1)[1] = "Individual"
names(pca19_inv9)[1] = "Individual"
names(pca22_inv11)[1] = "Individual"
names(pca29_inv21)[1] = "Individual"

# Import the beagle file from which heterozygosities will be calculated
beagle6_inv1 = read.beagle("<Beagle_File_used_to_Calculate_Covariance_Matrix>", nsnp=990, header=TRUE)
beagle19_inv9 = read.beagle("<Beagle_File_used_to_Calculate_Covariance_Matrix>", nsnp=241, header=TRUE)
beagle22_inv11 = read.beagle("<Beagle_File_used_to_Calculate_Covariance_Matrix>", nsnp=517, header=TRUE)
beagle29_inv21 = read.beagle("<Beagle_File_used_to_Calculate_Covariance_Matrix>", nsnp=555, header=TRUE)

# Calculate the heterozygosity from the beagle files
het6_inv1 = row.summary(beagle6_inv1)
het19_inv9 = row.summary(beagle19_inv9)
het22_inv11 = row.summary(beagle22_inv11)
het29_inv21 = row.summary(beagle29_inv21)

# Add the names of the individuals
het6_inv1 = cbind(individuals, het6_inv1)
het19_inv9 = cbind(individuals, het19_inv9)
het22_inv11 = cbind(individuals, het22_inv11)
het29_inv21 = cbind(individuals, het29_inv21)

# Keep only the needed information
het6_inv1 = data.frame(het6_inv1$V1, het6_inv1$Heterozygosity)
het19_inv9 = data.frame(het19_inv9$V1, het19_inv9$Heterozygosity)
het22_inv11 = data.frame(het22_inv11$V1, het22_inv11$Heterozygosity)
het29_inv21 = data.frame(het29_inv21$V1, het29_inv21$Heterozygosity)

# Change the name of the column with individuals
names(het6_inv1)[1] = "Individual"
names(het19_inv9)[1] = "Individual"
names(het22_inv11)[1] = "Individual"
names(het29_inv21)[1] = "Individual"

# Merge PCA dataframe and heterozygosity dataframe by individual
finalplot6_inv1 = merge(pca6_inv1, het6_inv1, by="Individual")
finalplot19_inv9 = merge(pca19_inv9, het19_inv9, by="Individual")
finalplot22_inv11 = merge(pca22_inv11, het22_inv11, by="Individual")
finalplot29_inv21 = merge(pca29_inv21, het29_inv21, by="Individual")

#### Plots ####

# Final plots of heterozygosity in inversions
plothet6_inv1 = ggplot(data=finalplot6_inv1, aes(x=`pca6_inv1.V1`, y=`het6_inv1.Heterozygosity`, colour=`pca6_inv1.Concrete_Location`)) +
  geom_point(aes(shape=`pca6_inv1.Type`), size=5, alpha=0.8) +
  ggtitle("Chr. 1 - Cluster 6") +
  xlab(paste0("PC1", sep=" (", round((eigen_data6_inv1$values[1]/sum(eigen_data6_inv1$values)*100),2), "%)")) +
  ylab(expression(H[O])) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_colour_manual(values=c("Adriatic_Sea"="orange", "Alaska"="darkolivegreen4", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="yellowgreen", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4"),
                      labels=c("Adriatic_Sea"="Adriatic Sea", "Alaska"="Alaska", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea")) +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

plothet19_inv9 = ggplot(data=finalplot19_inv9, aes(x=`pca19_inv9.V1`, y=`het19_inv9.Heterozygosity`, colour=`pca19_inv9.Concrete_Location`)) +
  geom_point(aes(shape=`pca19_inv9.Type`), size=5, alpha=0.8) +
  ggtitle("Chr. 9 - Cluster 19") +
  xlab(paste0("PC1", sep=" (", round((eigen_data19_inv9$values[1]/sum(eigen_data19_inv9$values)*100),2), "%)")) +
  ylab(expression(H[O])) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_colour_manual(values=c("Adriatic_Sea"="orange", "Alaska"="darkolivegreen4", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="yellowgreen", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4"),
                      labels=c("Adriatic_Sea"="Adriatic Sea", "Alaska"="Alaska", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea")) +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

plothet22_inv11 = ggplot(data=finalplot22_inv11, aes(x=`pca22_inv11.V1`, y=`het22_inv11.Heterozygosity`, colour=`pca22_inv11.Concrete_Location`)) +
  geom_point(aes(shape=`pca22_inv11.Type`), size=5, alpha=0.8) +
  ggtitle("Chr. 11 - Cluster 22") +
  xlab(paste0("PC1", sep=" (", round((eigen_data22_inv11$values[1]/sum(eigen_data22_inv11$values)*100),2), "%)")) +
  ylab(expression(H[O])) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_colour_manual(values=c("Adriatic_Sea"="orange", "Alaska"="darkolivegreen4", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="yellowgreen", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4"),
                      labels=c("Adriatic_Sea"="Adriatic Sea", "Alaska"="Alaska", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea")) +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

plothet29_inv21 = ggplot(data=finalplot29_inv21, aes(x=`pca29_inv21.V1`, y=`het29_inv21.Heterozygosity`, colour=`pca29_inv21.Concrete_Location`)) +
  geom_point(aes(shape=`pca29_inv21.Type`), size=5, alpha=0.8) +
  ggtitle("Chr. 21 - Cluster 29") +
  xlab(paste0("PC1", sep=" (", round((eigen_data29_inv21$values[1]/sum(eigen_data29_inv21$values)*100),2), "%)")) +
  ylab(expression(H[O])) +
  scale_shape_manual(values=c("Freshwater"=16, "Marine"=1)) +
  scale_colour_manual(values=c("Adriatic_Sea"="orange", "Alaska"="darkolivegreen4", "North_Scandinavia"="royalblue3", "East_Russia"="tomato4", "English_Channel"="yellowgreen", "Finnish_Golf"="purple3", "Iberian_Peninsula"="red", "North_Sea"="turquoise4"),
                      labels=c("Adriatic_Sea"="Adriatic Sea", "Alaska"="Alaska", "North_Scandinavia"="Fennoscandia", "East_Russia"="East Russia", "English_Channel"="English Channel", "Finnish_Golf"="Gulf of Finland", "Iberian_Peninsula"="Iberian Peninsula", "North_Sea"="North Sea")) +
  guides(colour=guide_legend(order=1, title="Geographic Region", override.aes=list(size=3, alpha=1)), shape=guide_legend(order=2, title="Habitat Type",override.aes=list(size=3, alpha=1))) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

png("Chromosomes 1, 9, 11, 21.png", units="in", width=7, height=6, res=900)
ggarrange(plothet6_inv1,
          plothet22_inv11,
          plothet29_inv21,
          plothet19_inv9,
          ncol=2, nrow=2, labels="auto", hjust=-1, align="hv", common.legend = TRUE, legend="bottom")
dev.off()
