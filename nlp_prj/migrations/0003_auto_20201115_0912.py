# Generated by Django 3.1.2 on 2020-11-15 02:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nlp_prj', '0002_auto_20201115_0812'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='article',
            name='title',
        ),
        migrations.AddField(
            model_name='article',
            name='top_1_accuracy',
            field=models.CharField(default='top', max_length=100),
        ),
    ]
