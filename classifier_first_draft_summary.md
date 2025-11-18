To give a short recap of the project I am pursuing:

1. I am testing to see if I can predict each Pokemon's "Experience Group" based on their other attributes.
2. There are 6 different experience groups, with 2 groups having less than 25 Pokemon each. (The groups are "Fluctuating", "Erratic", "Slow", "Medium Slow", "Medium Fast", "Fast")
3. The dataset I am working with is inherently small (~800) as there are only so many Pokemon that exist.

For this initial classifier, I chose to use Random Foresting. I did test out other algorithms in *otherClassifierTesting.py* but I found that regular old Random Foresting performed slightly better than ExtraTrees, XGBoost, CatBoost, and LightGBM. The same methods were used between classifiers, but Random Foresting generally just works better, likely due to the fact that the dataset is just small, and boosting algorithms usually want to have larger datasets. 

As for the features used in the classifier, I used, *base_total* (The sum of the Pokemon's base stats), *evolution_stages* (How many times a Pokemon can evolve from its base form), *color* (The color of the Pokemon), *capture_rate* (A multiplier used to determine how easy or hard it is to capture a Pokemon), *percentage_male* (The percentage chance a Pokemon will be Male), *base_experience* (How much expierience the Pokemon gives upon fainting), *base_egg_steps* (The amount of steps required to hatch an egg of that species), *no_in_generation* (How far the Pokemon is in that specific generation)

Previously on the *eda_summary.md* I mentioned that I was highly considering dropping the experience groups Fluctuating and Erratic due to their very low data count. It turns out that actually keeping them in the evaluation ever so slightly increased the accuracy of the model. This is especially interesting to note as these two experience groups were never predicted by the model, but still improved the overall accuracy.

Moving on to the model itself, one other thing I noticed about the data is that Pokemon can never switch experience group upon evolution. This essentially means that all Pokemon in an evolution line will all have the exact same experience group. (Eg Bulbasaur, Ivysaur, and Venusaur must ALL have the same experience grouping). Knowing this, I can ensure that upon predicting the experience group for the final evolution stage for a Pokemon, I can backfill the previous evolution stages too with the same exact experience group. The main reason I use the final evolution stage is that it shows the Pokemon at the pinnacle of their power, which gives the most accurate representation of their overall power. For example, if I just looked at Magikarp and Gyarados (In case you don't recognize these Pokemon, Magikarp is one of the weakest Pokemon ever but evolves into a very powerful Pokemon called Gyarados), it would easy for the classifer to put these Pokemon into different groups for obvious reasons, even though they both belong to the "Slow" group. Backfilling based on the most powerful evolution allows the classifier to more accurately look at a Pokemon's "potential", which is much more important for discovering the experience groupings.

Another thing to note, is that because the dataset is so small, it allows me to run iterations of the classifier quite quickly. Each run takes about 15 seconds overall, so its easy to make changes and iterate. Because of this, I chose to run 7 total different test-train splits --- one for each generation. Essentially, test on 1 generation, and train on the other 6. This allows me to see as well where the classifier fails, and what traits it may be lacking in. 

Below is the final PR curves for each group over every generation.
The baseline line is a basic random picker, which chooses a random experience group every time.`

![alt text](eval_with_all_groups_png/precision_recall_overall.png)

(If you would like to view the curves per-generation, or ROC curves, they are under the folder eval_with_all_groups_png)

The micro-average curve in pink is essentially the overall over all experience groups. The Erratic and Fluctuating are near 0 for obvious reasons - the classifier never chose them and as a result the PR is near 0. It makes sense for these to be under a random baseline as obviously the classifier never guessed these. Fast has a PR of 0.485, which makes sense. Given that this group has only a bit over 50 Pokemon overall, it's easy to see that there's not really much data to work off of. Still, there are valid patterns that can identify Pokemon in this group, but with the amount of data, its harder to pick up on these patterns. Medium Fast has a PR of 0.742, which is quite good actually, followed by Slow with 0.803 and Medium Slow with 0.848. It's quite easy to see that there is a trend with the amount of Pokemon there is in the experience group and the PR of that group. The overall PR sits at 0.760, which I would say is quite good overall.

In terms of data (# of Pokemon) from highest to lowest the ordering is:

Medium Fast (41.82%), Medium Slow (25.22%), Slow (21.47%), Fast (6.99%), Erratic (2.75%), Fluctuating (1.75%)

While the list in terms of PR is:

Medium Slow (0.848), Slow (0.803), Medium Fast (0.742), Fast (0.485), Erratic (0.080), Fluctuating (0.017)

Below is the overall accuracy breakdown of the experience groups. Acc represents the accuracy of each group (that is the % of the amount correct).

=== Overall (All Generations) ===
Total size: 801
Model: correct=570, incorrect=231, accuracy=0.712
Per true experience type (overall):
    Erratic         support=  22  correct=   0  incorrect=  22  acc= 0.000
    Fast            support=  56  correct=  17  incorrect=  39  acc= 0.304
    Fluctuating     support=  14  correct=   0  incorrect=  14  acc= 0.000
    Medium Fast     support= 335  correct= 288  incorrect=  47  acc= 0.860
    Medium Slow     support= 202  correct= 155  incorrect=  47  acc= 0.767
    Slow            support= 172  correct= 110  incorrect=  62  acc= 0.640
  Predicted counts (overall): Erratic=0, Fast=29, Fluctuating=0, Medium Fast=431, Medium Slow=190, Slow=151

Again, In terms of data (# of Pokemon) from highest to lowest the ordering is:

Medium Fast (41.82%), Medium Slow (25.22%), Slow (21.47%), Fast (6.99%), Erratic (2.75%), Fluctuating (1.75%)

While the list in terms of Accuracy is:

Medium Fast (0.860), Medium Slow (0.767), Slow (0.640), Fast (0.304), Erratic (0.000), Fluctuating (0.000)

You may notice that these exactly mirror each other. That is, from this we can observe that the larger percentage of Pokemon of an experience group there are, the better the prediction can be for that grouping. Quite literally the more data there is about the group, the better it does at predicting it, which makes sense. 

In terms of future improvements, I think the best improvements to make to the model would likely be to do some hyperparameter tuning for pretty obvious reasons, just to get better scores, but also to engineer more features / try out new or remove existing features. I'm currently working with 8 features in total right now, with many other fields in my parquet file being untouched. For example, the classifier right now doesn't account for a Pokemon's typing, abilities, weight, if they are legendary or not, height, the method in which they evolve, etc. These are all potentially useful features to improve the overall accuracy of the model but I haven't been able to test out yet. Also features such as a Pokemon's color may be not as helpful as I previously thought. Previously, I saw a larger proportion of Pokemon from then Fast experience group were Pink, so I though perhaps it would be a useful signal for that grouping. But I also failed to see how it affects the prediction of other groups and may just be amplifying noise in those other groups. Essentially, I will have to do a lot of testing with my parameters for the final evaluation. Another thing I can do to improve the accuracy is to simply get more data here. Data for Pokemon generations 8 and 9 do exist, but as far as I could tell, there's no dataset out there that has this information. Additionally, it would be a pain to do this myself for every Pokemon in these generations. 

Overall, I think reguardless this initial classifier was very succesful. 71.2% accuracy is honestly higher than I thought it would be. It should be realized as well that the data here was not observed in nature, but created by a bunch of Japanese people in an office, putting Pokemon into these experience groups for their silly game about catching monsters. I doubt they put in a crazy amount of thought into which Pokemon goes into which experience group, but they still did have some patterns that can be followed to predict which Pokemon belong to which groups. There are also some cases of obvious patterns being broken too (Such Mew, Celebi, and Shaymin being Legendary Pokemon in the Medium Slow group, DESPITE all other Legendary Pokemon being in the Slow group). But the fact that I can predict which one of these groups a Pokemon can fall into over 70% of the time is really neat. 