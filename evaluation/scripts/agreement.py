from sklearn.metrics import cohen_kappa_score

correct_paula = [4,4,3,4,3,3,4,3,4,3,3]
correct_amaia = [4,3,2,3,3,3,3,1,4,3,2]
correct_gpt = [5,4,4,3,4,4,4,3,5,4,4]


natural_paula = [3,2,2,4,3,2,3,1,4,2,2]
natural_amaia = [3,2,2,3,3,2,3,1,3,3,2]
natural_gpt = [4,4,3,3,4,3,3,2,4,4,4]

kappa_p_a = cohen_kappa_score(correct_paula, correct_amaia, labels=[1,2,3,4,5])
kappa_p_g = cohen_kappa_score(correct_paula, correct_gpt, labels=[1,2,3,4,5])
kappa_a_g = cohen_kappa_score(correct_amaia, correct_gpt, labels=[1,2,3,4,5])

kappa_natural_p_a = cohen_kappa_score(natural_paula, natural_amaia, labels=[1,2,3,4,5])
kappa_natural_p_g = cohen_kappa_score(natural_paula, natural_gpt, labels=[1,2,3,4,5])
kappa_natural_a_g = cohen_kappa_score(natural_amaia, natural_gpt, labels=[1,2,3,4,5])

print(kappa_p_a)
print(kappa_p_g)
print(kappa_a_g)
print(kappa_natural_p_a)
print(kappa_natural_p_g)
print(kappa_natural_a_g)