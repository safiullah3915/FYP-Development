from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0015_startup_embedding_needs_update_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="useronboardingpreferences",
            name="investor_profile",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]


