# Slurm submission scripts

When creating or updating a Slurm submission script, include these directives
in its initial `#SBATCH` header unless the user explicitly requests otherwise:

```bash
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
```
