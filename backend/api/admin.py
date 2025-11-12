from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Startup, StartupTag, Position, Application


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Admin configuration for User model"""
    list_display = ('username', 'email', 'role', 'is_active', 'is_staff', 'email_verified', 'created_at')
    list_filter = ('is_active', 'email_verified', 'is_staff', 'is_superuser', 'role')
    search_fields = ('username', 'email')
    ordering = ('-created_at',)
    
    # Update fieldsets to include role and phone_number
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('username', 'role', 'phone_number')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'email_verified', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'created_at', 'updated_at')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'password1', 'password2', 'role', 'is_staff', 'is_superuser'),
        }),
    )
    
    readonly_fields = ('created_at', 'updated_at', 'last_login')
    
    # Important: Tell admin to use email for authentication
    def get_fieldsets(self, request, obj=None):
        if not obj:
            return self.add_fieldsets
        return super().get_fieldsets(request, obj)


@admin.register(Startup)
class StartupAdmin(admin.ModelAdmin):
    """Admin configuration for Startup model"""
    list_display = ('title', 'owner', 'type', 'category', 'status', 'views', 'created_at')
    list_filter = ('type', 'category', 'status', 'featured', 'created_at')
    search_fields = ('title', 'description', 'field')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'views', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'role_title', 'description', 'field', 'website_url')
        }),
        ('Financial Information', {
            'fields': ('revenue', 'profit', 'asking_price', 'ttm_revenue', 'ttm_profit', 
                      'last_month_revenue', 'last_month_profit')
        }),
        ('Collaboration Information', {
            'fields': ('earn_through', 'phase', 'team_size')
        }),
        ('Classification', {
            'fields': ('type', 'category', 'status', 'featured')
        }),
        ('Metadata', {
            'fields': ('owner', 'stages', 'views', 'created_at', 'updated_at')
        }),
    )


@admin.register(StartupTag)
class StartupTagAdmin(admin.ModelAdmin):
    """Admin configuration for StartupTag model"""
    list_display = ('startup', 'tag')
    list_filter = ('tag',)
    search_fields = ('startup__title', 'tag')


@admin.register(Position)
class PositionAdmin(admin.ModelAdmin):
    """Admin configuration for Position model"""
    list_display = ('title', 'startup', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('title', 'startup__title')
    ordering = ('-created_at',)


@admin.register(Application)
class ApplicationAdmin(admin.ModelAdmin):
    """Admin configuration for Application model"""
    list_display = ('applicant', 'startup', 'position', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('applicant__username', 'startup__title', 'position__title')
    ordering = ('-created_at',)
    readonly_fields = ('created_at', 'updated_at')


