#!/bin/bash
################################################################################
# Pre-Transfer Verification Script
#
# Purpose: Verify prerequisites before running data transfer from Leonardo to MN5
# Run on: Leonardo
# Usage: bash verify_transfer_setup.sh
#
################################################################################

set -euo pipefail

# Configuration
MN5_USER="${MN5_USER:?set MN5_USER}"
MN5_HOST="login.mn5.bsc.es"
LEONARDO_DATA_SOURCE="${EUHPC_WORK:-/leonardo_work/EUHPC_E04_042}/datasets/PretrainDatasets"
MN5_DEST="/gpfs/scratch/ehpc678/leonardo_data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

################################################################################
# Check Functions
################################################################################

check_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((CHECKS_FAILED++))
}

check_warning() {
    echo -e "${YELLOW}⚠ WARNING${NC}: $1"
    ((CHECKS_WARNING++))
}

print_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_summary() {
    echo ""
    echo -e "${BLUE}=== VERIFICATION SUMMARY ===${NC}"
    echo -e "${GREEN}Passed:${NC}  ${CHECKS_PASSED}"
    echo -e "${RED}Failed:${NC}  ${CHECKS_FAILED}"
    echo -e "${YELLOW}Warnings:${NC} ${CHECKS_WARNING}"
    echo ""
}

################################################################################
# Main Verification
################################################################################

echo -e "${BLUE}Leonardo to MN5 Transfer - Pre-Verification Script${NC}"
echo "=================================================="
echo "Time: $(date)"
echo ""

# 1. Check Required Commands
print_section "Required Commands"

if command -v rsync &> /dev/null; then
    check_pass "rsync is installed ($(rsync --version | head -1 | cut -d' ' -f1-2))"
else
    check_fail "rsync is not installed"
    echo -e "         Load with: ${BLUE}module load rsync${NC}"
fi

if command -v ssh &> /dev/null; then
    check_pass "ssh is available"
else
    check_fail "ssh is not available"
fi

if command -v find &> /dev/null; then
    check_pass "find command is available"
else
    check_fail "find command is not available"
fi

# 2. Check Leonardo Source Data
print_section "Leonardo Source Data"

if [ -d "${LEONARDO_DATA_SOURCE}" ]; then
    check_pass "Source directory exists: ${LEONARDO_DATA_SOURCE}"
    
    # Check for specific datasets
    if [ -d "${LEONARDO_DATA_SOURCE}/iter-4" ]; then
        local iter4_size=$(du -sh "${LEONARDO_DATA_SOURCE}/iter-4" 2>/dev/null | cut -f1)
        check_pass "iter-4 dataset found (size: ~${iter4_size})"
    else
        check_warning "iter-4 dataset not found"
    fi
    
    if [ -d "${LEONARDO_DATA_SOURCE}/gl-2b" ]; then
        check_pass "gl-2b (Galician) dataset found"
    else
        check_warning "gl-2b (Galician) dataset not found"
    fi
    
    if [ -d "${LEONARDO_DATA_SOURCE}/ca-2b" ]; then
        check_pass "ca-2b (Catalan) dataset found"
    else
        check_warning "ca-2b (Catalan) dataset not found"
    fi
else
    check_fail "Source directory does not exist: ${LEONARDO_DATA_SOURCE}"
fi

# 3. Check SSH Setup
print_section "SSH Configuration"

# Check SSH key exists
if [ -f ~/.ssh/id_rsa ]; then
    check_pass "SSH private key exists (~/.ssh/id_rsa)"
else
    check_warning "SSH private key not found - may need to generate one"
    echo -e "         Generate with: ${BLUE}ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa${NC}"
fi

if [ -f ~/.ssh/id_rsa.pub ]; then
    check_pass "SSH public key exists (~/.ssh/id_rsa.pub)"
else
    check_warning "SSH public key not found"
fi

# Test SSH connectivity
if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes "${MN5_USER}@${MN5_HOST}" "echo 'SSH connection test'" > /dev/null 2>&1; then
    check_pass "SSH connection to ${MN5_USER}@${MN5_HOST} successful"
else
    check_fail "Cannot connect to ${MN5_USER}@${MN5_HOST} via SSH"
    echo -e "         Try: ${BLUE}ssh-copy-id -i ~/.ssh/id_rsa.pub ${MN5_USER}@${MN5_HOST}${NC}"
    echo -e "         Or:  ${BLUE}ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa${NC}"
fi

# 4. Check MN5 Destination
print_section "MN5 Destination Setup"

# Check if destination exists/writable
if ssh -o ConnectTimeout=5 "${MN5_USER}@${MN5_HOST}" "[ -w '${MN5_DEST}' ] || mkdir -p '${MN5_DEST}'" 2>/dev/null; then
    check_pass "MN5 destination directory accessible: ${MN5_DEST}"
else
    check_warning "Cannot verify MN5 destination - may need manual setup"
fi

# Check available space on MN5
if command_result=$(ssh -o ConnectTimeout=5 "${MN5_USER}@${MN5_HOST}" "df -B1 '${MN5_DEST}' 2>/dev/null | awk 'NR==2 {print \$4}'" 2>/dev/null); then
    if [ -n "${command_result}" ] && [ "${command_result}" -gt 0 ]; then
        available_gb=$((command_result / 1024 / 1024 / 1024))
        if [ "${available_gb}" -ge 500 ]; then
            check_pass "Adequate disk space on MN5: ${available_gb}GB available"
        elif [ "${available_gb}" -ge 100 ]; then
            check_warning "Limited disk space on MN5: ${available_gb}GB available (need 500GB+ for full transfer)"
        else
            check_fail "Insufficient disk space on MN5: only ${available_gb}GB available"
        fi
    fi
fi

# 5. Check Transfer Script
print_section "Transfer Script"

TRANSFER_SCRIPT="$HOME/mLatxa/datasets/export/transfer_leonardo_to_mn5.sh"
if [ -f "${TRANSFER_SCRIPT}" ]; then
    check_pass "Transfer script found: ${TRANSFER_SCRIPT}"
    
    if [ -x "${TRANSFER_SCRIPT}" ]; then
        check_pass "Transfer script is executable"
    else
        check_warning "Transfer script is not executable"
        echo -e "         Make executable with: ${BLUE}chmod +x ${TRANSFER_SCRIPT}${NC}"
    fi
else
    check_fail "Transfer script not found: ${TRANSFER_SCRIPT}"
fi

# 6. Check Configuration Files
print_section "MN5 Configuration Files"

CONFIG_DIR="$HOME/mLatxa/datasets/export/configs_mn5"
if [ -d "${CONFIG_DIR}" ]; then
    check_pass "Configuration directory found: ${CONFIG_DIR}"
    
    for config in v1_multimodal.json v2_multimodal.json multi.json; do
        if [ -f "${CONFIG_DIR}/${config}" ]; then
            check_pass "Config file found: ${config}"
        else
            check_warning "Config file not found: ${config}"
        fi
    done
else
    check_warning "Configuration directory not found: ${CONFIG_DIR}"
fi

# 7. Network Connectivity
print_section "Network Connectivity"

if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    check_pass "Internet connectivity available"
else
    check_warning "Internet connectivity check failed (may be normal in restricted environment)"
fi

if timeout 5 ssh -o ConnectTimeout=3 "${MN5_USER}@${MN5_HOST}" "ping -c 1 -W 2 8.8.8.8" > /dev/null 2>&1; then
    check_pass "MN5 has internet connectivity"
else
    check_warning "MN5 internet connectivity check failed"
fi

# Print Summary
print_summary

# Final Recommendation
echo -e "${BLUE}=== RECOMMENDATIONS ===${NC}"
if [ ${CHECKS_FAILED} -eq 0 ]; then
    echo -e "${GREEN}✓ System is ready for data transfer!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run dry-run to verify transfer:"
    echo -e "   ${BLUE}bash ${TRANSFER_SCRIPT} --dry-run${NC}"
    echo ""
    echo "2. Run actual transfer (iter-4 only - recommended first):"
    echo -e "   ${BLUE}bash ${TRANSFER_SCRIPT} --iter4-only${NC}"
    echo ""
    echo "3. Monitor progress:"
    echo -e "   ${BLUE}tail -f ~/data_transfer_logs/transfer_*.log${NC}"
else
    echo -e "${RED}✗ Please resolve the FAILED checks above before starting transfer${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. SSH key setup:"
    echo -e "   ${BLUE}ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa${NC}"
    echo -e "   ${BLUE}ssh-copy-id -i ~/.ssh/id_rsa.pub ${MN5_USER}@${MN5_HOST}${NC}"
    echo ""
    echo "2. Verify MN5 hostname (try these alternatives if ${MN5_HOST} doesn't work):"
    echo -e "   ${BLUE}mn5.bsc.es${NC}"
    echo -e "   ${BLUE}login.mn5.bsc.es${NC}"
    echo ""
    echo "3. Check rsync installation:"
    echo -e "   ${BLUE}module load rsync${NC}"
fi

# Exit with appropriate code
echo ""
if [ ${CHECKS_FAILED} -gt 0 ]; then
    echo "Verification complete: ${CHECKS_FAILED} critical issue(s) to resolve"
    exit 1
else
    echo "Verification complete: All critical checks passed"
    exit 0
fi
