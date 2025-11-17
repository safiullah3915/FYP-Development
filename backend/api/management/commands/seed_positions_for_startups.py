from django.core.management.base import BaseCommand
from django.db import transaction
from api.models import Startup, Position
from django.utils import timezone
import random


DEFAULT_TITLES = [
	"Full-Stack Developer",
	"Frontend Developer",
	"Backend Developer",
	"Mobile App Developer",
	"ML Engineer",
	"Data Engineer",
	"DevOps Engineer",
	"Product Designer",
	"Product Manager",
	"QA Engineer"
]

DEFAULT_REQUIREMENTS = [
	"Proficiency with JavaScript/TypeScript and React.",
	"Experience with Python and Django REST Framework.",
	"Familiarity with cloud services (AWS/GCP/Azure).",
	"Strong understanding of databases and SQL.",
	"Hands-on experience with CI/CD and Docker.",
	"Knowledge of modern CSS frameworks (Tailwind/Chakra).",
	"Experience with version control (Git) and code reviews.",
	"Ability to work in a startup environment and iterate quickly."
]


class Command(BaseCommand):
	help = "Create at least one active position for every collaboration startup (idempotent)."

	def add_arguments(self, parser):
		parser.add_argument(
			'--all-types',
			action='store_true',
			help='Also seed positions for marketplace startups (default seeds only collaboration).'
		)
		parser.add_argument(
			'--per-startup',
			type=int,
			default=1,
			help='Number of positions to create per startup that has none active (default: 1).'
		)

	@transaction.atomic
	def handle(self, *args, **options):
		include_all_types = options['all_types']
		num_per_startup = max(1, min(5, options['per_startup']))

		startups_qs = Startup.objects.filter(status='active')
		if not include_all_types:
			startups_qs = startups_qs.filter(type='collaboration')

		total_startups = startups_qs.count()
		created_total = 0
		skipped = 0

		self.stdout.write(self.style.MIGRATE_HEADING(f"Seeding positions for {total_startups} startups"))

		for s in startups_qs:
			active_count = s.positions.filter(is_active=True).count()
			if active_count > 0:
				skipped += 1
				continue

			for i in range(num_per_startup):
				title = random.choice(DEFAULT_TITLES)
				req = " ".join(random.sample(DEFAULT_REQUIREMENTS, k=min(3, len(DEFAULT_REQUIREMENTS))))
				Position.objects.create(
					startup=s,
					title=title,
					description=(s.description[:240] + '...') if s.description else f"{title} role at {s.title}",
					requirements=req,
					is_active=True,
					created_at=timezone.now()
				)
				created_total += 1

		self.stdout.write(self.style.SUCCESS(f"Created {created_total} active positions; skipped {skipped} (already had active)."))
		self.stdout.write(self.style.SUCCESS("Done."))

